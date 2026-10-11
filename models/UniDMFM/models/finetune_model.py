"""EEG-Arena fine-tuning wrapper for the official Uni-DMFM architecture."""
from __future__ import annotations

import os

import torch
import torch.nn as nn

from ..config import UniDMFMConfig
from .upstream.model import UniDMFM


class LinearWithConstraint(nn.Linear):
    """Author-released downstream linear head, including its initial max norm."""

    def __init__(self, *args, max_norm=1.0, flatten=0, dropout=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.flatten = int(flatten)
        self.drop_out = nn.Dropout(float(dropout)) if dropout else None
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=float(max_norm))

    def forward(self, x):
        if self.flatten:
            x = x.flatten(self.flatten)
        if self.drop_out is not None:
            x = self.drop_out(x)
        return super().forward(x)


class Model(nn.Module):
    """Unified Uni-DMFM classifier returning ``[B, classes]`` logits."""

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.variant = getattr(params, "variant", None) or UniDMFMConfig.DEFAULT_VARIANT
        if self.variant not in UniDMFMConfig.WEIGHT_VARIANTS:
            raise ValueError(f"Unknown Uni-DMFM variant: {self.variant}")
        self.num_channels = int(getattr(params, "num_channels", 1))
        self.patch_num = int(getattr(params, "patch_num", 4))
        if self.num_channels < 1 or self.num_channels > 256:
            raise ValueError("Uni-DMFM supports between 1 and 256 channels")
        if self.patch_num < 1 or self.patch_num > 100:
            raise ValueError("Uni-DMFM supports between 1 and 100 200-sample patches")
        self.backbone = UniDMFM()
        self.classifier = LinearWithConstraint(
            self.num_channels * self.patch_num * 384,
            int(getattr(params, "num_of_classes", 2)),
            max_norm=1,
            flatten=1,
        )

        if getattr(params, "use_pretrained_weights", False):
            path = self._resolve_weight_path(
                getattr(params, "foundation_dir", UniDMFMConfig.DEFAULT_WEIGHT_PATH)
            )
            self._load_pretrained(path)
        if getattr(params, "frozen", False):
            self.freeze_backbone()

    def _resolve_weight_path(self, path):
        if os.path.isfile(path):
            return path
        return os.path.join(path, UniDMFMConfig.WEIGHT_VARIANTS[self.variant]["file"])

    def _load_pretrained(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Uni-DMFM checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        raw_state = checkpoint.get("model", checkpoint)
        clean_state = {
            key: value
            for key, value in raw_state.items()
            if "total_ops" not in key and "total_params" not in key
        }
        self.backbone.load_state_dict(clean_state, strict=True)
        print(f"[Uni-DMFM] Loaded {len(clean_state)} backbone tensors from {path}")

    def _validate_input(self, x):
        if x.ndim != 4 or x.shape[-1] != 200:
            raise ValueError(f"Uni-DMFM expects [B,C,P,200], got {tuple(x.shape)}")
        if x.shape[1] != self.num_channels or x.shape[2] != self.patch_num:
            raise ValueError(
                "Uni-DMFM downstream head was initialized for "
                f"C={self.num_channels}, P={self.patch_num}, got C={x.shape[1]}, P={x.shape[2]}"
            )
        return x.contiguous()

    def extract_features(self, x):
        features = self.backbone(self._validate_input(x), mode="finetune")
        return features.flatten(1)

    def forward(self, x):
        return self.classifier(self.extract_features(x))

    def freeze_backbone(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        print("[Uni-DMFM] Backbone frozen; constrained task head remains trainable")

    def get_param_groups(self):
        return [
            {
                "params": [p for p in self.backbone.parameters() if p.requires_grad],
                "lr_scale": float(getattr(self.params, "encoder_lr_scale", 1.0)),
            },
            {
                "params": [p for p in self.classifier.parameters() if p.requires_grad],
                "lr_scale": float(getattr(self.params, "classifier_lr_scale", 10.0)),
            },
        ]
