import torch
import torch.nn as nn

from models._shared import LinearWithConstraint, RawBaselineModelBase
from .backbone import ShallowConvNetBackbone


class Model(RawBaselineModelBase):
    """ShallowConvNet wrapper that follows the classic shallow FBCSP-inspired design."""

    def __init__(self, params):
        super().__init__(params)
        num_classes = getattr(params, "num_of_classes", 2)
        num_channels = getattr(params, "num_channels", 64)
        samples = getattr(params, "samples", 1000)
        n_filters_time = getattr(params, "n_filters_time", 40)
        n_filters_spat = getattr(params, "n_filters_spat", 40)
        filter_time_length = getattr(params, "filter_time_length", 25)
        pool_time_length = getattr(params, "pool_time_length", 75)
        pool_time_stride = getattr(params, "pool_time_stride", 15)
        dropout = getattr(params, "dropout", 0.5)

        self.backbone = ShallowConvNetBackbone(
            num_channels=num_channels,
            n_times=samples,
            n_filters_time=n_filters_time,
            n_filters_spat=n_filters_spat,
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
            raise ValueError(f"ShallowConvNet expects raw input [B, C, T], got shape {tuple(x.shape)}")
        x = x.unsqueeze(1)
        features = self.backbone(x)
        return torch.flatten(features, start_dim=1)
