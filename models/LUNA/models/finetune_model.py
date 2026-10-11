"""LUNA fine-tuning wrapper for the unified EEG framework."""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import LUNAConfig
from .backbone import LUNABackbone


class Model(nn.Module):
    """Unified LUNA model where `forward(x)` returns logits."""

    def __init__(self, params):
        super().__init__()
        self.params = params

        variant = getattr(params, "variant", None) or LUNAConfig.DEFAULT_VARIANT
        if variant not in LUNAConfig.WEIGHT_VARIANTS:
            available = ", ".join(LUNAConfig.WEIGHT_VARIANTS.keys())
            raise ValueError(f"Unknown LUNA variant '{variant}'. Available: {available}")

        variant_cfg = LUNAConfig.WEIGHT_VARIANTS[variant]
        self.patch_size = int(variant_cfg["patch_size"])

        num_classes = int(getattr(params, "num_of_classes", 2))
        classifier_dropout = float(getattr(params, "classifier_dropout", 0.15))

        self.backbone = LUNABackbone(
            patch_size=self.patch_size,
            num_queries=int(variant_cfg["num_queries"]),
            embed_dim=int(variant_cfg["embed_dim"]),
            depth=int(variant_cfg["depth"]),
            num_heads=int(variant_cfg["num_heads"]),
            mlp_ratio=float(variant_cfg["mlp_ratio"]),
            num_classes=num_classes,
            classifier_dropout=classifier_dropout,
        )

        if getattr(params, "use_pretrained_weights", False):
            foundation_dir = getattr(
                params,
                "foundation_dir",
                "./models/Weights/LUNA/LUNA_base.safetensors",
            )
            pretrained_file = self._resolve_pretrained_path(foundation_dir, variant)
            self._load_pretrained_weights(pretrained_file)

        if getattr(params, "frozen", False):
            self.freeze_backbone()

    def _resolve_pretrained_path(self, foundation_dir, variant):
        if os.path.isfile(foundation_dir):
            return foundation_dir

        if os.path.isdir(foundation_dir):
            candidate = os.path.join(
                foundation_dir,
                LUNAConfig.WEIGHT_VARIANTS[variant]["file"],
            )
            if os.path.exists(candidate):
                return candidate

            fallback = os.path.join(foundation_dir, "LUNA_base.safetensors")
            if os.path.exists(fallback):
                return fallback
            return candidate

        return foundation_dir

    @staticmethod
    def _extract_state_dict(checkpoint):
        if not isinstance(checkpoint, dict):
            return checkpoint

        for key in ("state_dict", "model", "model_state_dict", "net"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        return checkpoint

    @staticmethod
    def _normalize_key(key):
        normalized = key
        for prefix in ("module.", "model.", "backbone.", "net."):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
        return normalized

    def _load_pretrained_weights(self, pretrained_file):
        if not os.path.exists(pretrained_file):
            print(f"[LUNA] Warning: pretrained weights not found at {pretrained_file}")
            return

        if pretrained_file.endswith(".safetensors"):
            from safetensors.torch import load_file

            state_dict = load_file(pretrained_file)
        else:
            checkpoint = torch.load(pretrained_file, map_location="cpu", weights_only=False)
            state_dict = self._extract_state_dict(checkpoint)

        normalized = {
            self._normalize_key(key): value
            for key, value in state_dict.items()
            if isinstance(value, torch.Tensor)
        }

        model_state = self.backbone.state_dict()
        matched = {
            key: value
            for key, value in normalized.items()
            if key in model_state and model_state[key].shape == value.shape
        }
        mismatched = [
            key
            for key, value in normalized.items()
            if key in model_state and model_state[key].shape != value.shape
        ]

        msg = self.backbone.load_state_dict(matched, strict=False)
        print(f"[LUNA] Loaded pretrained weights from {pretrained_file}")
        print(f"[LUNA] Matched {len(matched)}/{len(normalized)} keys")
        if mismatched:
            print(f"[LUNA] Skipped (shape mismatch): {mismatched[:5]}... (total {len(mismatched)})")
        if msg.missing_keys:
            print(f"[LUNA] Missing keys: {msg.missing_keys[:5]}... (total {len(msg.missing_keys)})")
        if msg.unexpected_keys:
            print(
                f"[LUNA] Unexpected keys: {msg.unexpected_keys[:5]}... "
                f"(total {len(msg.unexpected_keys)})"
            )

    def _prepare_input(self, x):
        if x.dim() == 4:
            batch, channels, steps, patch = x.shape
            x = x.reshape(batch, channels, steps * patch)
        elif x.dim() != 3:
            raise ValueError(f"LUNA expects 3D or 4D EEG input, got shape {tuple(x.shape)}")

        time_len = x.shape[-1]
        pad_len = (self.patch_size - (time_len % self.patch_size)) % self.patch_size
        if pad_len > 0:
            x = F.pad(x, (0, pad_len))
        return x

    def _build_channel_locations(self, x):
        batch, channels, _ = x.shape
        ch_positions = getattr(self.params, "ch_positions", None)
        ch_names = list(getattr(self.params, "ch_names", []) or [])

        locs = []
        for idx in range(channels):
            channel_name = ch_names[idx] if idx < len(ch_names) else None
            xyz = [0.0, 0.0, 0.0]
            if isinstance(ch_positions, dict) and channel_name is not None:
                raw = ch_positions.get(channel_name)
                if raw is None:
                    raw = ch_positions.get(f"EEG:{channel_name}")
                if raw is not None:
                    values = list(raw)
                    if len(values) < 3:
                        values = values + [0.0] * (3 - len(values))
                    xyz = [float(values[0]), float(values[1]), float(values[2])]
            locs.append(xyz)

        loc_tensor = torch.tensor(locs, dtype=x.dtype, device=x.device)
        loc_tensor = loc_tensor.unsqueeze(0).expand(batch, -1, -1)
        return loc_tensor

    def extract_features(self, x):
        x = self._prepare_input(x)
        channel_locations = self._build_channel_locations(x)
        return self.backbone.extract_features(x, channel_locations, mask=None)

    def forward(self, x):
        x = self._prepare_input(x)
        channel_locations = self._build_channel_locations(x)
        logits, _ = self.backbone(x, mask=None, channel_locations=channel_locations)
        return logits

    def freeze_backbone(self):
        for name, param in self.backbone.named_parameters():
            if name.startswith("classifier."):
                continue
            param.requires_grad = False
        print("[LUNA] Backbone frozen (classifier remains trainable)")

    def get_param_groups(self):
        encoder_lr_scale = float(getattr(self.params, "encoder_lr_scale", 1.0))
        classifier_lr_scale = float(getattr(self.params, "classifier_lr_scale", 10.0))

        backbone_params = []
        classifier_params = []
        for name, param in self.backbone.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("classifier."):
                classifier_params.append(param)
            else:
                backbone_params.append(param)

        return [
            {"params": backbone_params, "lr_scale": encoder_lr_scale},
            {"params": classifier_params, "lr_scale": classifier_lr_scale},
        ]
