import torch
import torch.nn as nn

from models._shared import LinearWithConstraint, RawBaselineModelBase

from .backbone import EEGNeXBackbone


class Model(RawBaselineModelBase):
    """EEGNeX wrapper that keeps the official EEGNeX_8_32 topology."""

    def __init__(self, params):
        super().__init__(params)
        num_classes = getattr(params, "num_of_classes", 2)
        num_channels = getattr(params, "num_channels", 64)
        samples = getattr(params, "samples", 1000)
        stem_filters = getattr(params, "stem_filters", 8)
        mid_filters = getattr(params, "mid_filters", 32)
        depth_multiplier = getattr(params, "depth_multiplier", 2)
        kernel_length = getattr(params, "kernel_length", 64)
        dilated_kernel_length = getattr(params, "dilated_kernel_length", 16)
        first_dilation = getattr(params, "first_dilation", 2)
        second_dilation = getattr(params, "second_dilation", 4)
        pool_size = getattr(params, "pool_size", 4)
        dropout = getattr(params, "dropout", 0.5)

        self.backbone = EEGNeXBackbone(
            num_channels=num_channels,
            n_times=samples,
            stem_filters=stem_filters,
            mid_filters=mid_filters,
            depth_multiplier=depth_multiplier,
            kernel_length=kernel_length,
            dilated_kernel_length=dilated_kernel_length,
            first_dilation=first_dilation,
            second_dilation=second_dilation,
            pool_size=pool_size,
            dropout=dropout,
        )
        self.classifier = LinearWithConstraint(
            self.backbone.feature_dim,
            num_classes,
            max_norm=0.25,
        )

    def extract_features(self, x):
        if x.ndim != 3:
            raise ValueError(f"EEGNeX expects raw input [B, C, T], got shape {tuple(x.shape)}")
        x = x.unsqueeze(1)
        features = self.backbone(x)
        return torch.flatten(features, start_dim=1)
