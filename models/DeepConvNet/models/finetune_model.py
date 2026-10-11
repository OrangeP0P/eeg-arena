import torch
import torch.nn as nn

from models._shared import LinearWithConstraint, RawBaselineModelBase
from .backbone import DeepConvNetBackbone


class Model(RawBaselineModelBase):
    """DeepConvNet wrapper that stays close to the classic implementation."""

    def __init__(self, params):
        super().__init__(params)
        num_classes = getattr(params, "num_of_classes", 2)
        num_channels = getattr(params, "num_channels", 64)
        samples = getattr(params, "samples", 1000)
        n_filters_time = getattr(params, "n_filters_time", 25)
        filter_time_length = getattr(params, "filter_time_length", 10)
        pool_time_length = getattr(params, "pool_time_length", 3)
        pool_time_stride = getattr(params, "pool_time_stride", 3)
        dropout = getattr(params, "dropout", 0.5)

        self.backbone = DeepConvNetBackbone(
            num_channels=num_channels,
            n_times=samples,
            n_filters_time=n_filters_time,
            filter_time_length=filter_time_length,
            pool_time_length=pool_time_length,
            pool_time_stride=pool_time_stride,
            dropout=dropout,
        )
        self.classifier = LinearWithConstraint(
            self.backbone.feature_dim,
            num_classes,
            max_norm=0.5,
        )

    def extract_features(self, x):
        if x.ndim != 3:
            raise ValueError(f"DeepConvNet expects raw input [B, C, T], got shape {tuple(x.shape)}")
        x = x.unsqueeze(1)
        features = self.backbone(x)
        return torch.flatten(features, start_dim=1)
