import torch
import torch.nn as nn

from models._shared import RawBaselineModelBase

from .backbone import EEGInceptionERPBackbone


class Model(RawBaselineModelBase):
    """EEGInceptionERP wrapper that keeps the official ERP EEG-Inception topology."""

    def __init__(self, params):
        super().__init__(params)
        num_classes = getattr(params, "num_of_classes", 2)
        num_channels = getattr(params, "num_channels", 64)
        samples = getattr(params, "samples", 1000)
        filters_per_branch = getattr(params, "filters_per_branch", 8)
        depth_multiplier = getattr(params, "depth_multiplier", 2)
        kernel_lengths = (
            getattr(params, "kernel_length_long", 64),
            getattr(params, "kernel_length_medium", 32),
            getattr(params, "kernel_length_short", 16),
        )
        dropout = getattr(params, "dropout", 0.25)
        activation = getattr(params, "activation", "elu")

        self.backbone = EEGInceptionERPBackbone(
            num_channels=num_channels,
            n_times=samples,
            filters_per_branch=filters_per_branch,
            depth_multiplier=depth_multiplier,
            kernel_lengths=kernel_lengths,
            dropout=dropout,
            activation=activation,
        )
        self.classifier = nn.Linear(self.backbone.feature_dim, num_classes)

    def extract_features(self, x):
        if x.ndim != 3:
            raise ValueError(f"EEGInceptionERP expects raw input [B, C, T], got shape {tuple(x.shape)}")
        x = x.transpose(1, 2).unsqueeze(1)
        features = self.backbone(x)
        return torch.flatten(features, start_dim=1)
