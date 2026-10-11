"""MSVTNet fine-tuning wrapper for the unified EEG framework."""

import torch
import torch.nn as nn

from models._shared import RawBaselineModelBase

from .backbone import MSVTNetBackbone


def _parse_int_list(raw_value, arg_name):
    if isinstance(raw_value, str):
        values = [item.strip() for item in raw_value.split(",")]
    elif isinstance(raw_value, (list, tuple)):
        values = list(raw_value)
    else:
        raise TypeError(f"{arg_name} must be a comma-separated string or a list of ints.")

    parsed = []
    for value in values:
        if value == "":
            continue
        parsed.append(int(value))

    if not parsed:
        raise ValueError(f"{arg_name} must contain at least one integer value.")
    return parsed


class Model(RawBaselineModelBase):
    """Unified MSVTNet wrapper where `forward(x)` returns final logits."""

    def __init__(self, params):
        super().__init__(params)
        num_classes = getattr(params, "num_of_classes", 2)
        num_channels = getattr(params, "num_channels", 64)
        samples = getattr(params, "samples", 1000)

        branch_filters = _parse_int_list(getattr(params, "branch_filters", "9,9,9,9"), "branch_filters")
        branch_kernel_sizes = _parse_int_list(
            getattr(params, "branch_kernel_sizes", "15,31,63,125"),
            "branch_kernel_sizes",
        )
        temporal_kernel_size = getattr(params, "temporal_kernel_size", 15)
        depth_multiplier = getattr(params, "depth_multiplier", 2)
        pool_size1 = getattr(params, "pool_size1", 8)
        pool_size2 = getattr(params, "pool_size2", 7)
        cnn_dropout = getattr(params, "cnn_dropout", 0.3)
        attention_heads = getattr(params, "attention_heads", 8)
        ff_ratio = getattr(params, "ff_ratio", 1.0)
        transformer_dropout = getattr(params, "transformer_dropout", 0.5)
        transformer_layers = getattr(params, "transformer_layers", 2)

        self.backbone = MSVTNetBackbone(
            num_channels=num_channels,
            n_times=samples,
            branch_filters=branch_filters,
            branch_kernel_sizes=branch_kernel_sizes,
            temporal_kernel_size=temporal_kernel_size,
            depth_multiplier=depth_multiplier,
            pool_size1=pool_size1,
            pool_size2=pool_size2,
            cnn_dropout=cnn_dropout,
            attention_heads=attention_heads,
            ff_ratio=ff_ratio,
            transformer_dropout=transformer_dropout,
            transformer_layers=transformer_layers,
        )
        self.classifier = nn.Linear(self.backbone.feature_dim, num_classes)
        self.branch_classifiers = nn.ModuleList(
            [nn.Linear(feature_dim, num_classes) for feature_dim in self.backbone.branch_feature_dims]
        )
        self._cached_branch_logits = None

    @staticmethod
    def _validate_input(x):
        if x.ndim != 3:
            raise ValueError(f"MSVTNet expects raw input [B, C, T], got shape {tuple(x.shape)}")
        return x.unsqueeze(1)

    def _forward_all_heads(self, x):
        x = self._validate_input(x)
        global_feature, branch_features = self.backbone(x)
        logits = self.classifier(global_feature)
        branch_logits = [
            head(feature)
            for head, feature in zip(self.branch_classifiers, branch_features)
        ]
        return global_feature, logits, branch_logits

    def extract_features(self, x):
        global_feature, _, _ = self._forward_all_heads(x)
        return global_feature

    def forward(self, x):
        _, logits, branch_logits = self._forward_all_heads(x)
        self._cached_branch_logits = branch_logits
        return logits

    def compute_loss(self, logits, targets, criterion):
        main_loss = criterion(logits, targets)
        if not getattr(self.params, "enable_auxiliary_branch_loss", True):
            return main_loss

        branch_logits = self._cached_branch_logits
        if not branch_logits:
            return main_loss

        branch_loss_lambda = float(getattr(self.params, "branch_loss_lambda", 0.6))
        if not 0.0 <= branch_loss_lambda <= 1.0:
            raise ValueError("branch_loss_lambda must be between 0 and 1.")

        auxiliary_loss = torch.stack([criterion(branch_logit, targets) for branch_logit in branch_logits]).sum()
        return branch_loss_lambda * main_loss + (1.0 - branch_loss_lambda) * auxiliary_loss

    def get_param_groups(self):
        encoder_lr_scale = getattr(self.params, "encoder_lr_scale", 1.0)
        classifier_lr_scale = getattr(self.params, "classifier_lr_scale", 1.0)
        head_params = list(self.classifier.parameters()) + list(self.branch_classifiers.parameters())
        return [
            {"params": list(self.backbone.parameters()), "lr_scale": encoder_lr_scale},
            {"params": head_params, "lr_scale": classifier_lr_scale},
        ]
