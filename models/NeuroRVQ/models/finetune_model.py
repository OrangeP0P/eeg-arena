"""NeuroRVQ fine-tuning wrapper for the unified EEG framework."""
import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.channels import MODEL_CHANNEL_SPACE_NEURORVQ

from .backbone import NeuroRVQBackbone, NeuroRVQBackboneConfig


def _canonical_channel_key(name):
    text = str(name).replace("EEG:", "").strip()
    return re.sub(r"[^A-Za-z0-9]", "", text).upper()


class Model(nn.Module):
    """Unified NeuroRVQ model wrapper (forward returns logits)."""

    def __init__(self, params):
        super().__init__()
        self.params = params

        self.patch_size = int(getattr(params, "patch_size", 200))
        self.n_patches = int(getattr(params, "n_patches", 256))
        self.channel_space = list(MODEL_CHANNEL_SPACE_NEURORVQ)
        self.channel_lookup = {
            _canonical_channel_key(name): idx for idx, name in enumerate(self.channel_space)
        }

        backbone_config = NeuroRVQBackboneConfig(
            n_patches=self.n_patches,
            patch_size=self.patch_size,
            out_chans=int(getattr(params, "out_chans", 8)),
            embed_dim=int(getattr(params, "embed_dim", 200)),
            depth=int(getattr(params, "depth", 12)),
            num_heads=int(getattr(params, "num_heads", 10)),
            mlp_ratio=float(getattr(params, "mlp_ratio", 4.0)),
            qkv_bias=bool(getattr(params, "qkv_bias", True)),
            drop_rate=float(getattr(params, "drop_rate", 0.0)),
            attn_drop_rate=float(getattr(params, "attn_drop_rate", 0.0)),
            drop_path_rate=float(getattr(params, "drop_path_rate", 0.0)),
            init_values=float(getattr(params, "init_values", 1e-5)),
            init_scale=float(getattr(params, "init_scale", 0.001)),
            n_global_electrodes=len(self.channel_space),
        )
        self.backbone = NeuroRVQBackbone(backbone_config)

        if getattr(params, "use_pretrained_weights", False):
            foundation_dir = getattr(
                params,
                "foundation_dir",
                "./models/Weights/NeuroRVQ/NeuroRVQ_EEG_foundation_model_v1.pt",
            )
            pretrained_file = self._resolve_pretrained_path(foundation_dir)
            self._load_pretrained_weights(pretrained_file)

        feature_dim = int(backbone_config.embed_dim) * 4
        num_classes = int(getattr(params, "num_of_classes", 2))
        hidden_dim = int(getattr(params, "classifier_hidden_dim", 0))
        dropout = float(getattr(params, "dropout", 0.1))

        if hidden_dim > 0:
            self.classifier = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Dropout(dropout),
                nn.Linear(feature_dim, num_classes),
            )

        if getattr(params, "frozen", False):
            self.freeze_backbone()

    def _resolve_pretrained_path(self, foundation_dir):
        if os.path.isfile(foundation_dir):
            return foundation_dir

        if os.path.isdir(foundation_dir):
            for candidate in (
                "NeuroRVQ_EEG_foundation_model_v1.pt",
                "model.pth",
                "checkpoint.pth",
            ):
                candidate_path = os.path.join(foundation_dir, candidate)
                if os.path.exists(candidate_path):
                    return candidate_path
            return os.path.join(foundation_dir, "NeuroRVQ_EEG_foundation_model_v1.pt")

        return foundation_dir

    def _extract_state_dict(self, checkpoint):
        if isinstance(checkpoint, dict):
            if "model" in checkpoint and isinstance(checkpoint["model"], dict):
                return checkpoint["model"]
            if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
                return checkpoint["state_dict"]
        return checkpoint

    def _normalize_state_dict(self, state_dict):
        normalized = {}
        for key, value in state_dict.items():
            clean_key = key
            for prefix in ("module.", "backbone."):
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix):]
            normalized[clean_key] = value
        return normalized

    def _load_pretrained_weights(self, pretrained_file):
        if not os.path.exists(pretrained_file):
            print(f"[NeuroRVQ] Warning: pretrained weights not found at {pretrained_file}")
            return

        checkpoint = torch.load(pretrained_file, map_location="cpu", weights_only=False)
        state_dict = self._extract_state_dict(checkpoint)
        state_dict = self._normalize_state_dict(state_dict)

        backbone_state = self.backbone.state_dict()
        matched = {
            key: value
            for key, value in state_dict.items()
            if isinstance(value, torch.Tensor)
            and key in backbone_state
            and backbone_state[key].shape == value.shape
        }
        skipped = [
            key
            for key, value in state_dict.items()
            if isinstance(value, torch.Tensor)
            and key in backbone_state
            and backbone_state[key].shape != value.shape
        ]

        msg = self.backbone.load_state_dict(matched, strict=False)
        print(f"[NeuroRVQ] Loaded pretrained weights from {pretrained_file}")
        print(f"[NeuroRVQ] Matched {len(matched)}/{len(state_dict)} keys")
        if skipped:
            print(f"[NeuroRVQ] Skipped (shape mismatch): {skipped[:5]}... (total {len(skipped)})")
        if msg.missing_keys:
            print(f"[NeuroRVQ] Missing keys: {msg.missing_keys[:5]}... (total {len(msg.missing_keys)})")
        if msg.unexpected_keys:
            print(f"[NeuroRVQ] Unexpected keys: {msg.unexpected_keys[:5]}... (total {len(msg.unexpected_keys)})")

    def _to_patched(self, x):
        if x.dim() == 3:
            batch, channels, time_len = x.shape
            pad_len = (self.patch_size - time_len % self.patch_size) % self.patch_size
            if pad_len > 0:
                x = F.pad(x, (0, pad_len))
                time_len = x.shape[-1]
            x = x.view(batch, channels, time_len // self.patch_size, self.patch_size)
        elif x.dim() != 4:
            raise ValueError(f"NeuroRVQ expects 3D or 4D EEG input, got shape {tuple(x.shape)}")

        if x.shape[-1] != self.patch_size:
            batch, channels, patches, patch_len = x.shape
            x = x.reshape(batch * channels * patches, 1, patch_len)
            x = F.interpolate(x, size=self.patch_size, mode="linear", align_corners=False)
            x = x.reshape(batch, channels, patches, self.patch_size)

        return x

    def _resolve_channel_indices(self, num_channels):
        model_channel_indices = getattr(self.params, "model_channel_indices", None)
        if isinstance(model_channel_indices, torch.Tensor):
            model_channel_indices = model_channel_indices.tolist()
        if model_channel_indices is not None:
            model_channel_indices = [int(idx) for idx in model_channel_indices]
            if len(model_channel_indices) == num_channels:
                return model_channel_indices

        ch_names = getattr(self.params, "ch_names", None)
        if ch_names is not None:
            mapped = []
            missing = []
            for name in list(ch_names)[:num_channels]:
                key = _canonical_channel_key(name)
                if key in self.channel_lookup:
                    mapped.append(self.channel_lookup[key])
                else:
                    missing.append(name)
            if missing:
                raise ValueError(
                    "NeuroRVQ channel mapping failed for channels: "
                    + ", ".join(map(str, missing[:5]))
                    + (" ..." if len(missing) > 5 else "")
                )
            if len(mapped) == num_channels:
                return mapped

        if num_channels > len(self.channel_space):
            raise ValueError(
                f"NeuroRVQ received {num_channels} channels, but channel space supports only {len(self.channel_space)}."
            )
        return list(range(num_channels))

    def _build_embedding_indices(self, num_channels, patch_num, device):
        if patch_num > self.n_patches:
            raise ValueError(
                f"NeuroRVQ got patch_num={patch_num}, but n_patches={self.n_patches}. "
                "Increase --n_patches or reduce sample length."
            )

        channel_indices = self._resolve_channel_indices(num_channels)

        temporal_embedding_ix = torch.arange(
            self.n_patches - patch_num,
            self.n_patches,
            device=device,
            dtype=torch.long,
        )
        temporal_embedding_ix = temporal_embedding_ix.repeat(num_channels).view(1, -1)

        spatial_embedding_ix = torch.tensor(channel_indices, device=device, dtype=torch.long)
        spatial_embedding_ix = spatial_embedding_ix.repeat_interleave(patch_num).view(1, -1)

        return temporal_embedding_ix, spatial_embedding_ix

    def extract_features(self, x):
        x = self._to_patched(x)
        _, num_channels, patch_num, _ = x.shape
        temporal_embedding_ix, spatial_embedding_ix = self._build_embedding_indices(
            num_channels=num_channels,
            patch_num=patch_num,
            device=x.device,
        )
        return self.backbone.forward_features(x, temporal_embedding_ix, spatial_embedding_ix)

    def forward(self, x):
        features = self.extract_features(x)
        return self.classifier(features)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("[NeuroRVQ] Backbone frozen")

    def get_param_groups(self):
        encoder_lr_scale = float(getattr(self.params, "encoder_lr_scale", 1.0))
        classifier_lr_scale = float(getattr(self.params, "classifier_lr_scale", 10.0))

        backbone_params = [param for param in self.backbone.parameters() if param.requires_grad]
        classifier_params = [param for param in self.classifier.parameters() if param.requires_grad]

        return [
            {"params": backbone_params, "lr_scale": encoder_lr_scale},
            {"params": classifier_params, "lr_scale": classifier_lr_scale},
        ]
