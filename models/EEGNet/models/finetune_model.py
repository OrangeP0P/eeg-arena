import torch
import torch.nn as nn

from models._shared import LinearWithConstraint, RawBaselineModelBase
from .backbone import EEGNetBackbone


class Model(RawBaselineModelBase):
    """EEGNet wrapper that stays close to the classic EEGNet-v4 implementation."""

    def __init__(self, params):
        super().__init__(params)
        num_classes = getattr(params, "num_of_classes", 2)
        num_channels = getattr(params, "num_channels", 64)
        samples = getattr(params, "samples", 1000)
        F1 = getattr(params, "F1", 8)
        depth_multiplier = getattr(params, "D", 2)
        kernel_length = getattr(params, "kernel_length", 64)
        dropout = getattr(params, "dropout", 0.25)

        self.backbone = EEGNetBackbone(
            num_channels=num_channels,
            n_times=samples,
            F1=F1,
            D=depth_multiplier,
            kernel_length=kernel_length,
            dropout=dropout,
        )
        self.classifier = LinearWithConstraint(
            self.backbone.feature_dim,
            num_classes,
            max_norm=0.25,
        )

    def extract_features(self, x):
        if x.ndim != 3:
            raise ValueError(f"EEGNet expects raw input [B, C, T], got shape {tuple(x.shape)}")
        x = x.unsqueeze(1)
        features = self.backbone(x)
        return torch.flatten(features, start_dim=1)
