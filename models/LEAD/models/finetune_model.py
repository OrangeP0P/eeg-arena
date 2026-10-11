"""Unified LEAD fine-tuning wrapper."""

import os
from collections import OrderedDict

import torch
import torch.nn as nn

from .backbone import LEADv2Backbone


class Model(nn.Module):
    """LEAD wrapper with the unified fine-tuning interface."""

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.sampling_rate = int(getattr(params, "sampling_rate", 200))

        self.backbone = LEADv2Backbone(params)

        num_classes = int(getattr(params, "num_of_classes", 2))
        self.classifier = nn.Linear(self.backbone.feature_dim, num_classes)

        if getattr(params, "use_pretrained_weights", False):
            checkpoint_path = self._resolve_checkpoint_path(
                getattr(params, "foundation_dir", "./models/Weights/LEAD/p-base")
            )
            self._load_pretrained_weights(checkpoint_path)

        if getattr(params, "frozen", False):
            self.freeze_backbone()

    def _resolve_checkpoint_path(self, foundation_dir):
        if os.path.isfile(foundation_dir):
            return foundation_dir
        return os.path.join(foundation_dir, "checkpoint.pth")

    def _extract_state_dict(self, checkpoint):
        if not isinstance(checkpoint, dict):
            return checkpoint

        nested_dict = None
        for key in ("state_dict", "model", "model_state_dict"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict):
                nested_dict = candidate
                break
        if nested_dict is None:
            nested_dict = checkpoint

        normalized = OrderedDict()
        for key, value in nested_dict.items():
            if key == "n_averaged":
                continue

            while key.startswith("module."):
                key = key[len("module."):]
            if key.startswith("backbone."):
                key = key[len("backbone."):]

            normalized[key] = value

        return normalized

    def _load_pretrained_weights(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            print(f"[LEAD] Warning: pretrained weights not found at {checkpoint_path}")
            return

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = self._extract_state_dict(checkpoint)
        msg = self.backbone.load_state_dict(state_dict, strict=False)
        print(f"[LEAD] Loaded pretrained weights from {checkpoint_path}")
        if msg.missing_keys:
            print(f"[LEAD] Missing keys: {msg.missing_keys[:5]}... (total {len(msg.missing_keys)})")
        if msg.unexpected_keys:
            print(f"[LEAD] Unexpected keys: {msg.unexpected_keys[:5]}... (total {len(msg.unexpected_keys)})")

    def _sampling_rate_tokens(self, batch_size, device):
        return torch.full((batch_size,), self.sampling_rate, dtype=torch.long, device=device)

    def extract_features(self, x):
        if x.ndim != 3:
            raise ValueError(f"LEAD expects raw input [B, C, T], got shape {tuple(x.shape)}")
        x = x.transpose(1, 2).contiguous()
        fs = self._sampling_rate_tokens(x.size(0), x.device)
        return self.backbone.extract_features(x, fs=fs)

    def forward(self, x):
        features = self.extract_features(x)
        return self.classifier(features)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("[LEAD] Backbone frozen")

    def get_param_groups(self):
        encoder_lr_scale = getattr(self.params, "encoder_lr_scale", 1.0)
        classifier_lr_scale = getattr(self.params, "classifier_lr_scale", 10.0)
        backbone_params = [param for param in self.backbone.parameters() if param.requires_grad]
        classifier_params = [param for param in self.classifier.parameters() if param.requires_grad]
        return [
            {"params": backbone_params, "lr_scale": encoder_lr_scale},
            {"params": classifier_params, "lr_scale": classifier_lr_scale},
        ]
