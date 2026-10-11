"""
EEGMamba wrapper that adapts the upstream backbone to the unified interface.
"""
import os

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from .eegmamba import EEGMamba


class Model(nn.Module):
    """
    Unified EEGMamba finetune model.

    Input:
        patched EEG of shape (B, C, S, 200)
    Output:
        logits of shape (B, num_classes)
    """

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.embed_dim = 200

        num_classes = getattr(params, "num_of_classes", 2)
        num_channels = getattr(params, "num_channels", 64)
        patch_num = getattr(params, "patch_num", 30)
        dropout = getattr(params, "dropout", 0.1)

        self.backbone = EEGMamba(
            patch_size=200,
            d_model=self.embed_dim,
            n_layer=12,
            ssm_headdim=50,
            ssm_d_state=64,
        )

        if getattr(params, "use_pretrained_weights", False):
            foundation_dir = getattr(
                params,
                "foundation_dir",
                "./models/Weights/EEGMamba/pretrained_EEGMamba.pth",
            )
            pretrained_file = self._resolve_pretrained_path(foundation_dir)

            if os.path.exists(pretrained_file):
                checkpoint = torch.load(pretrained_file, map_location="cpu", weights_only=False)
                if "model" in checkpoint:
                    state_dict = checkpoint["model"]
                elif "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                else:
                    state_dict = checkpoint

                normalized_state_dict = {}
                for key, value in state_dict.items():
                    new_key = key.replace("module.", "").replace("backbone.", "")
                    normalized_state_dict[new_key] = value

                msg = self.backbone.load_state_dict(normalized_state_dict, strict=False)
                print(f"[EEGMamba] Loaded pretrained weights from {pretrained_file}")
                if msg.missing_keys:
                    print(f"[EEGMamba] Missing keys: {msg.missing_keys[:5]}... (total {len(msg.missing_keys)})")
                if msg.unexpected_keys:
                    print(
                        f"[EEGMamba] Unexpected keys: {msg.unexpected_keys[:5]}... "
                        f"(total {len(msg.unexpected_keys)})"
                    )
            else:
                print(f"[EEGMamba] Warning: Pretrained weights not found at {pretrained_file}")

        self.backbone.proj_out = nn.Identity()
        self.classifier = self._build_classifier(
            classifier_type=getattr(params, "classifier", "all_patch_reps"),
            num_channels=num_channels,
            patch_num=patch_num,
            num_classes=num_classes,
            dropout=dropout,
        )

        if getattr(params, "frozen", False):
            self.freeze_backbone()

    def _resolve_pretrained_path(self, foundation_dir):
        if not os.path.isabs(foundation_dir):
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            foundation_dir = os.path.join(project_root, foundation_dir)

        if os.path.isdir(foundation_dir):
            return os.path.join(foundation_dir, "pretrained_EEGMamba.pth")
        return foundation_dir

    def _build_classifier(self, classifier_type, num_channels, patch_num, num_classes, dropout):
        flatten_dim = num_channels * patch_num * self.embed_dim
        classifier_hidden = max(4, patch_num) * self.embed_dim

        if classifier_type == "avgpooling_patch_reps":
            return nn.Sequential(
                Rearrange("b c s d -> b d c s"),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(self.embed_dim, num_classes),
            )
        if classifier_type == "all_patch_reps_onelayer":
            return nn.Sequential(
                Rearrange("b c s d -> b (c s d)"),
                nn.Linear(flatten_dim, num_classes),
            )
        if classifier_type == "all_patch_reps_twolayer":
            return nn.Sequential(
                Rearrange("b c s d -> b (c s d)"),
                nn.Linear(flatten_dim, self.embed_dim),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(self.embed_dim, num_classes),
            )
        if classifier_type == "all_patch_reps":
            return nn.Sequential(
                Rearrange("b c s d -> b (c s d)"),
                nn.Linear(flatten_dim, classifier_hidden),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(classifier_hidden, self.embed_dim),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(self.embed_dim, num_classes),
            )
        raise ValueError(f"Unknown EEGMamba classifier type: {classifier_type}")

    def extract_features(self, x):
        """Return backbone patch representations."""
        return self.backbone(x)

    def forward(self, x):
        feats = self.extract_features(x)
        return self.classifier(feats)

    def freeze_backbone(self):
        """Freeze backbone parameters for linear probing style finetuning."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("[EEGMamba] Backbone frozen")

    def get_param_groups(self):
        """Return optimizer parameter groups for backbone and classifier."""
        encoder_lr_scale = getattr(self.params, "encoder_lr_scale", 1.0)
        classifier_lr_scale = getattr(self.params, "classifier_lr_scale", 10.0)
        return [
            {"params": list(self.backbone.parameters()), "lr_scale": encoder_lr_scale},
            {"params": list(self.classifier.parameters()), "lr_scale": classifier_lr_scale},
        ]
