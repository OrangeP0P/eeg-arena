"""ATCNet fine-tuning wrapper for the unified EEG framework."""
import torch
import torch.nn as nn

from models._shared import LinearWithConstraint
from .backbone import ATCNetBackbone


class ATCNetClassifier(nn.Module):
    """Classifier head that mirrors ATCNet's average/concat fusion modes."""

    def __init__(self, feature_dim, num_classes, n_windows, fuse="average"):
        super().__init__()
        self.fuse = fuse

        if fuse == "average":
            self.window_heads = nn.ModuleList(
                [
                    LinearWithConstraint(feature_dim, num_classes, max_norm=0.5)
                    for _ in range(n_windows)
                ]
            )
        elif fuse == "concat":
            self.concat_head = LinearWithConstraint(
                feature_dim * n_windows,
                num_classes,
                max_norm=0.5,
            )
        else:
            raise ValueError(f"Unsupported ATCNet fuse mode '{fuse}'.")

    def forward(self, window_features):
        if self.fuse == "average":
            logits = [
                head(window_features[:, window_index, :])
                for window_index, head in enumerate(self.window_heads)
            ]
            return torch.stack(logits, dim=0).mean(dim=0)

        return self.concat_head(window_features.flatten(start_dim=1))


class Model(nn.Module):
    """Unified ATCNet wrapper where `forward(x)` returns logits."""

    def __init__(self, params):
        super().__init__()
        self.params = params

        num_classes = getattr(params, "num_of_classes", 2)
        num_channels = getattr(params, "num_channels", 64)
        samples = getattr(params, "samples", 1000)

        n_windows = getattr(params, "n_windows", 5)
        attention = getattr(params, "attention", "mha")
        attention_heads = getattr(params, "attention_heads", 2)
        attention_key_dim = getattr(params, "attention_key_dim", 8)
        attention_dropout = getattr(params, "attention_dropout", 0.5)
        attention_residual_dropout = getattr(params, "attention_residual_dropout", 0.3)
        attention_se_ratio = getattr(params, "attention_se_ratio", 8)

        eegn_F1 = getattr(params, "eegn_F1", 16)
        eegn_D = getattr(params, "eegn_D", 2)
        eegn_kernel_size = getattr(params, "eegn_kernel_size", 64)
        eegn_pool_size = getattr(params, "eegn_pool_size", 7)
        eegn_dropout = getattr(params, "eegn_dropout", 0.3)

        tcn_depth = getattr(params, "tcn_depth", 2)
        tcn_kernel_size = getattr(params, "tcn_kernel_size", 4)
        tcn_filters = getattr(params, "tcn_filters", 32)
        tcn_dropout = getattr(params, "tcn_dropout", 0.3)
        tcn_activation = getattr(params, "tcn_activation", "elu")
        fuse = getattr(params, "fuse", "average")

        self.backbone = ATCNetBackbone(
            num_channels=num_channels,
            n_times=samples,
            n_windows=n_windows,
            attention=attention,
            attention_heads=attention_heads,
            attention_key_dim=attention_key_dim,
            attention_dropout=attention_dropout,
            attention_residual_dropout=attention_residual_dropout,
            attention_se_ratio=attention_se_ratio,
            eegn_F1=eegn_F1,
            eegn_D=eegn_D,
            eegn_kernel_size=eegn_kernel_size,
            eegn_pool_size=eegn_pool_size,
            eegn_dropout=eegn_dropout,
            tcn_depth=tcn_depth,
            tcn_kernel_size=tcn_kernel_size,
            tcn_filters=tcn_filters,
            tcn_dropout=tcn_dropout,
            tcn_activation=tcn_activation,
        )
        self.classifier = ATCNetClassifier(
            feature_dim=self.backbone.output_dim,
            num_classes=num_classes,
            n_windows=n_windows,
            fuse=fuse,
        )

        if getattr(params, "frozen", False):
            self.freeze_backbone()

    @staticmethod
    def _validate_input(x):
        if x.ndim != 3:
            raise ValueError(f"ATCNet expects raw input [B, C, T], got shape {tuple(x.shape)}")
        return x.unsqueeze(1)

    def forward(self, x):
        x = self._validate_input(x)
        window_features = self.backbone(x)
        return self.classifier(window_features)

    def extract_features(self, x):
        x = self._validate_input(x)
        window_features = self.backbone(x)
        if self.classifier.fuse == "concat":
            return window_features.flatten(start_dim=1)
        return window_features.mean(dim=1)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def get_param_groups(self):
        encoder_lr_scale = getattr(self.params, "encoder_lr_scale", 1.0)
        classifier_lr_scale = getattr(self.params, "classifier_lr_scale", 1.0)
        return [
            {"params": list(self.backbone.parameters()), "lr_scale": encoder_lr_scale},
            {"params": list(self.classifier.parameters()), "lr_scale": classifier_lr_scale},
        ]
