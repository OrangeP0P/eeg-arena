"""EEG-Arena wrapper around SleepGPT's released time-frequency encoder."""

from __future__ import annotations

import os
import re

import torch
from torch import nn
from torch.nn import functional as F

from .upstream_multiway_transformer import (
    backbone_base_patch200,
    backbone_huge_patch200,
    backbone_large_patch200,
)


SLEEP_CHANNELS = ("C3", "C4", "EMG", "EOG", "F3", "FPZ", "O1", "PZ")


def _normalise_channel(name: str) -> str:
    value = str(name).upper().replace("EEG:", "").replace("EEG ", "")
    return re.sub(r"[^A-Z0-9]", "", value)


class Model(nn.Module):
    """SleepGPT encoder plus an EEG-Arena task-specific classification head."""

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.arch = str(getattr(params, "sleepgpt_arch", "large"))
        num_classes = int(getattr(params, "num_of_classes", 2))
        dropout = float(getattr(params, "classifier_dropout", 0.1))

        cfg = {
            "drop_path_rate": 0.0,
            "random_choose_channels": len(SLEEP_CHANNELS),
            "actual_channels": "none",
            "mask_ratio": 0.75,
            "epoch_duration": 30,
            "fs": 100,
            "fft_only": False,
            "time_only": False,
            "loss_names": {"mtm": 1, "itc": 0, "itm": 0},
            "use_cb": True,
        }
        builders = {
            "base": backbone_base_patch200,
            "large": backbone_large_patch200,
            "huge": backbone_huge_patch200,
        }
        self.backbone = builders[self.arch](
            patch_size=200,
            config=cfg,
            use_relative_pos_emb=False,
            use_triton=False,
            spo2_ods_settings={},
        )
        # EEG-Arena calls ``embed(..., mask=False)``.  The released masked-token
        # parameter is therefore retained for checkpoint compatibility but is
        # not part of the downstream classification graph.
        self.backbone.mask_token.requires_grad_(False)
        self.token_type_embeddings = nn.Embedding(2, self.backbone.embed_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.backbone.embed_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(self.backbone.embed_dim * 2, num_classes),
        )
        self.pretrained_load_report = {"matched": 0, "available": 0}

        if getattr(params, "use_pretrained_weights", False):
            checkpoint = str(getattr(params, "foundation_dir", ""))
            if os.path.isdir(checkpoint):
                checkpoint = os.path.join(
                    checkpoint,
                    "ModelCheckpoint-epoch=79-val_acc=0.0000-val_score=4.2305.ckpt",
                )
            if self.arch != "large":
                print(
                    f"[SleepGPT] Official weights are only available for large; "
                    f"'{self.arch}' is initialized from scratch."
                )
            else:
                self._load_pretrained(checkpoint)

        if getattr(params, "frozen", False):
            self.freeze_backbone()

    @staticmethod
    def _state_dict(checkpoint):
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model", "model_state_dict"):
                if isinstance(checkpoint.get(key), dict):
                    return checkpoint[key]
        return checkpoint

    def _load_pretrained(self, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"SleepGPT checkpoint not found: {path}")
        payload = torch.load(
            path, map_location="cpu", weights_only=False, mmap=True
        )
        state = self._state_dict(payload)
        backbone_state = {}
        token_state = {}
        for raw_key, value in state.items():
            key = raw_key.removeprefix("module.")
            if key.startswith("transformer."):
                backbone_state[key[len("transformer.") :]] = value
            elif key.startswith("token_type_embeddings."):
                token_state[key[len("token_type_embeddings.") :]] = value

        # The public checkpoint contains the complete large backbone plus the
        # token-type embedding.  The only excluded tensors are the two masked
        # reconstruction decoders, which are not on the classification path.
        self.backbone.load_state_dict(backbone_state, strict=True)
        self.token_type_embeddings.load_state_dict(token_state, strict=True)
        expected = len(self.backbone.state_dict()) + len(
            self.token_type_embeddings.state_dict()
        )
        self.pretrained_load_report = {
            "matched": len(backbone_state) + len(token_state),
            "available": len(backbone_state) + len(token_state),
            "expected": expected,
            "missing": 0,
            "excluded_pretraining_decoder_tensors": len(state)
            - len(backbone_state)
            - len(token_state),
        }
        print(
            f"[SleepGPT] Strictly loaded {len(backbone_state) + len(token_state)}"
            f"/{expected} encoder tensors from {path}"
        )

    def _channel_indices(self, channels: int) -> list[int | None]:
        names = list(getattr(self.params, "ch_names", []) or [])
        normalised = [_normalise_channel(name) for name in names[:channels]]
        indices: list[int | None] = []
        for target in SLEEP_CHANNELS:
            target_norm = _normalise_channel(target)
            matches = []
            for index, name in enumerate(normalised):
                if target_norm in {"EMG", "EOG"}:
                    if target_norm in name:
                        matches.append(index)
                elif name == target_norm or name.startswith(target_norm + "A") or name.startswith(target_norm + "M"):
                    matches.append(index)
            indices.append(matches[0] if matches else None)
        return indices

    def _map_channels(self, x: torch.Tensor):
        batch, channels, time = x.shape
        indices = self._channel_indices(channels)
        if not any(index is not None for index in indices):
            raise ValueError(
                "SleepGPT found no supported channel. Expected one of "
                + ", ".join(SLEEP_CHANNELS)
            )
        mapped = x.new_zeros(batch, len(SLEEP_CHANNELS), time)
        present = torch.zeros(batch, len(SLEEP_CHANNELS), dtype=torch.bool, device=x.device)
        for target_index, source_index in enumerate(indices):
            if source_index is not None:
                mapped[:, target_index] = x[:, source_index]
                present[:, target_index] = True
        return mapped, present

    @staticmethod
    def _segments(x: torch.Tensor, length: int = 3000):
        if x.shape[-1] < length:
            return F.pad(x, (0, length - x.shape[-1])).unsqueeze(1)
        count = max(1, x.shape[-1] // length)
        return x[..., : count * length].reshape(x.shape[0], x.shape[1], count, length).permute(0, 2, 1, 3)

    def _encode_epoch(self, epoch: torch.Tensor, present: torch.Tensor):
        fft, fft_mask = self.backbone.get_fft(epoch, present.float(), aug=False)
        patches = present.repeat_interleave(self.backbone.num_patches, dim=1)
        embedded = self.backbone.embed(
            [epoch, fft], attn_mask=patches, mask=False
        )
        time_len = embedded["x_len"]
        values = embedded["x"]
        values = torch.cat(
            [
                values[:, :time_len]
                + self.token_type_embeddings(
                    torch.zeros(values.shape[0], time_len, dtype=torch.long, device=values.device)
                ),
                values[:, time_len:]
                + self.token_type_embeddings(
                    torch.ones(
                        values.shape[0], values.shape[1] - time_len,
                        dtype=torch.long, device=values.device,
                    )
                ),
            ],
            dim=1,
        )
        cls = torch.ones(values.shape[0], 1, dtype=torch.bool, device=values.device)
        attention_mask = torch.cat([cls, patches, cls, fft_mask.repeat_interleave(self.backbone.num_patches, dim=1)], dim=1)
        for block in self.backbone.blocks:
            values = block(values, mask=attention_mask, modality_type="tf")
        values = self.backbone.norm(values)
        return torch.cat([values[:, 0], values[:, time_len]], dim=-1)

    def extract_features(self, x: torch.Tensor):
        if x.ndim == 4:
            x = x.flatten(2)
        if x.ndim != 3:
            raise ValueError(f"SleepGPT expects [B,C,T], got {tuple(x.shape)}")
        mapped, present = self._map_channels(x)
        windows = self._segments(mapped)
        features = [self._encode_epoch(windows[:, index], present) for index in range(windows.shape[1])]
        return torch.stack(features, dim=1).mean(dim=1)

    def forward(self, x: torch.Tensor):
        return self.classifier(self.extract_features(x))

    def freeze_backbone(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for parameter in self.token_type_embeddings.parameters():
            parameter.requires_grad = False

    def get_param_groups(self):
        encoder = [p for p in list(self.backbone.parameters()) + list(self.token_type_embeddings.parameters()) if p.requires_grad]
        head = [p for p in self.classifier.parameters() if p.requires_grad]
        return [
            {"params": encoder, "lr_scale": float(getattr(self.params, "encoder_lr_scale", 1.0))},
            {"params": head, "lr_scale": float(getattr(self.params, "classifier_lr_scale", 10.0))},
        ]
