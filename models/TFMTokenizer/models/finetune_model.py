"""End-to-end official TFM-Tokenizer plus TFM-Encoder EEG-Arena adapter."""
from __future__ import annotations

import os

import torch
import torch.nn as nn

from ..config import TFMTokenizerConfig
from .upstream.tfm_token import (
    get_tfm_token_classifier_64x4,
    get_tfm_tokenizer_2x2x8,
)


class Model(nn.Module):
    """Tokenize 200-Hz raw EEG and return downstream classification logits."""

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.variant = getattr(params, "variant", None) or TFMTokenizerConfig.DEFAULT_VARIANT
        if self.variant not in TFMTokenizerConfig.WEIGHT_VARIANTS:
            raise ValueError(f"Unknown TFM-Tokenizer variant: {self.variant}")
        self.tokenizer = get_tfm_tokenizer_2x2x8(code_book_size=8192, emb_size=64)
        self.encoder = get_tfm_token_classifier_64x4(
            n_classes=int(getattr(params, "num_of_classes", 2)),
            code_book_size=8192,
            emb_size=64,
        )
        for parameter in self.tokenizer.parameters():
            parameter.requires_grad = False

        if getattr(params, "use_pretrained_weights", False):
            encoder_path = self._resolve_encoder_path(
                getattr(params, "foundation_dir", TFMTokenizerConfig.DEFAULT_WEIGHT_PATH)
            )
            tokenizer_path = self._resolve_tokenizer_path(
                getattr(params, "tfm_tokenizer_dir", None)
            )
            self._load_tokenizer(tokenizer_path)
            self._load_encoder(encoder_path)
        self.tokenizer.eval()
        if getattr(params, "frozen", False):
            self.freeze_backbone()

    def train(self, mode=True):
        super().train(mode)
        # The EMA quantizer mutates its codebook in train mode even under no_grad.
        self.tokenizer.eval()
        return self

    def _resolve_encoder_path(self, path):
        if os.path.isfile(path):
            return path
        relative = TFMTokenizerConfig.WEIGHT_VARIANTS[self.variant]["file"]
        return os.path.join(path, relative)

    def _resolve_tokenizer_path(self, override):
        if override:
            if os.path.isfile(override):
                return override
            relative = TFMTokenizerConfig.WEIGHT_VARIANTS[self.variant]["tokenizer_file"]
            return os.path.join(override, relative)
        relative = TFMTokenizerConfig.WEIGHT_VARIANTS[self.variant]["tokenizer_file"]
        return os.path.join(TFMTokenizerConfig.WEIGHTS_DIR, relative)

    def _load_tokenizer(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"TFM tokenizer checkpoint not found: {path}")
        state = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model_state = self.tokenizer.state_dict()
        # Early single-dataset releases use a single Linear reconstruction decoder,
        # whereas the pinned source uses a two-layer decoder. Tokenization never
        # executes this reconstruction-only module, so validate the complete encode
        # path strictly and load decoder tensors only when their shapes match.
        execution_keys = {key for key in model_state if not key.startswith("decoder.")}
        mismatched = [
            key
            for key, value in state.items()
            if key in execution_keys and model_state[key].shape != value.shape
        ]
        if mismatched:
            raise RuntimeError(f"TFM tokenizer execution-path shape mismatch: {mismatched[:5]}")
        matched = {
            key: value
            for key, value in state.items()
            if key in model_state and model_state[key].shape == value.shape
        }
        missing = sorted(execution_keys - set(matched))
        if missing:
            raise RuntimeError(
                "TFM tokenizer checkpoint is missing execution-path tensors: "
                + ", ".join(missing[:5])
            )
        self.tokenizer.load_state_dict(matched, strict=False)
        print(
            f"[TFM-Tokenizer] Loaded {len(execution_keys)}/{len(execution_keys)} "
            f"execution-path tensors ({len(matched)} total) from {path}"
        )

    def _load_encoder(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"TFM encoder checkpoint not found: {path}")
        state = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model_state = self.encoder.state_dict()
        mismatched = [
            key
            for key, value in state.items()
            if not key.startswith("classification_head.")
            and key in model_state
            and model_state[key].shape != value.shape
        ]
        if mismatched:
            raise RuntimeError(f"TFM encoder checkpoint shape mismatch: {mismatched[:5]}")
        matched = {
            key: value
            for key, value in state.items()
            if not key.startswith("classification_head.")
            and key in model_state
            and model_state[key].shape == value.shape
        }
        expected = {key for key in model_state if not key.startswith("classification_head.")}
        missing = sorted(expected - set(matched))
        if missing:
            raise RuntimeError("TFM checkpoint is missing encoder tensors: " + ", ".join(missing[:5]))
        self.encoder.load_state_dict(matched, strict=False)
        print(f"[TFM-Encoder] Loaded {len(matched)} non-task tensors from {path}")

    @staticmethod
    def _prepare_raw(x):
        if x.ndim == 4:
            batch, channels, patches, samples = x.shape
            x = x.reshape(batch, channels, patches * samples)
        if x.ndim != 3:
            raise ValueError(f"TFM-Tokenizer expects [B,C,T], got {tuple(x.shape)}")
        if x.shape[1] > 16:
            raise ValueError("The released TFM-Encoder has 16 channel embeddings")
        if x.shape[-1] < 200:
            raise ValueError("TFM-Tokenizer requires at least one second (200 samples) of EEG")
        return x

    def _tokenize(self, x):
        x = self._prepare_raw(x)
        batch, channels, samples = x.shape
        temporal = x.reshape(batch * channels, samples)
        window = torch.hann_window(200, dtype=x.dtype, device=x.device)
        spectrum = torch.stft(
            temporal,
            n_fft=200,
            hop_length=100,
            onesided=True,
            return_complex=True,
            center=False,
            window=window,
        ).abs()[:, :100]
        with torch.no_grad():
            _, indices, _ = self.tokenizer.tokenize(spectrum, temporal)
        time_tokens = indices.shape[-1]
        if time_tokens > 1000:
            raise ValueError("TFM-Encoder temporal positional encoding supports at most 1000 tokens/channel")
        if 1 + channels * time_tokens > 2048:
            raise ValueError(
                f"TFM-Encoder sequence limit is 2048 tokens, got {1 + channels * time_tokens}"
            )
        return indices.reshape(batch, channels, time_tokens)

    def _encode_tokens(self, indices):
        embedded = self.encoder.eeg_token_embedding(indices)
        channels = embedded.shape[1]
        channel_ids = self.encoder.index[:channels]
        channel_embedding = self.encoder.channel_embed(channel_ids)[None, :, None, :]
        embedded = embedded + channel_embedding
        batch, channels, time_tokens, dim = embedded.shape
        embedded = embedded.reshape(batch * channels, time_tokens, dim)
        embedded = self.encoder.temporal_pos_embed(embedded)
        embedded = embedded.reshape(batch, channels * time_tokens, dim)
        cls = self.encoder.cls_token.expand(batch, -1, -1)
        encoded = self.encoder.LAT(torch.cat([cls, embedded], dim=1))
        return encoded[:, 0]

    def extract_features(self, x):
        return self._encode_tokens(self._tokenize(x))

    def forward(self, x):
        return self.encoder.classification_head(self.extract_features(x))

    def freeze_backbone(self):
        for name, parameter in self.encoder.named_parameters():
            if not name.startswith("classification_head."):
                parameter.requires_grad = False
        print("[TFM-Tokenizer] Tokenizer and TFM-Encoder frozen; task head remains trainable")

    def get_param_groups(self):
        backbone = [
            parameter
            for name, parameter in self.encoder.named_parameters()
            if parameter.requires_grad and not name.startswith("classification_head.")
        ]
        head = [p for p in self.encoder.classification_head.parameters() if p.requires_grad]
        return [
            {
                "params": backbone,
                "lr_scale": float(getattr(self.params, "encoder_lr_scale", 1.0)),
            },
            {
                "params": head,
                "lr_scale": float(getattr(self.params, "classifier_lr_scale", 10.0)),
            },
        ]
