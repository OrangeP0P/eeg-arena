import torch
import torch.nn as nn

from models._shared.baseline_utils import Conv2dWithConstraint


class Square(nn.Module):
    """Element-wise square used in the classic shallow architecture."""

    def forward(self, x):
        return torch.square(x)


class SafeLog(nn.Module):
    """Numerically safe logarithm after average pooling."""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return torch.log(torch.clamp(x, min=self.eps))


class ShallowConvNetBackbone(nn.Module):
    """Shallow ConvNet feature extractor close to the classic FBCSP-inspired design."""

    def __init__(
        self,
        num_channels,
        n_times,
        n_filters_time=40,
        n_filters_spat=40,
        filter_time_length=25,
        pool_time_length=75,
        pool_time_stride=15,
        dropout=0.5,
    ):
        super().__init__()

        self.features = nn.Sequential(
            Conv2dWithConstraint(
                1,
                n_filters_time,
                kernel_size=(1, filter_time_length),
                bias=False,
                max_norm=2.0,
            ),
            Conv2dWithConstraint(
                n_filters_time,
                n_filters_spat,
                kernel_size=(num_channels, 1),
                bias=False,
                max_norm=2.0,
            ),
            nn.BatchNorm2d(n_filters_spat),
            Square(),
            nn.AvgPool2d(
                kernel_size=(1, pool_time_length),
                stride=(1, pool_time_stride),
            ),
            SafeLog(),
            nn.Dropout(dropout),
        )
        self.feature_dim = self._infer_feature_dim(num_channels=num_channels, n_times=n_times)

    def _infer_feature_dim(self, num_channels, n_times):
        with torch.no_grad():
            dummy = torch.ones(1, 1, num_channels, n_times)
            features = self.forward(dummy)
        return int(features.flatten(start_dim=1).shape[1])

    def forward(self, x):
        return self.features(x)
