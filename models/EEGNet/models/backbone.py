import torch
import torch.nn as nn
import torch.nn.functional as F

from models._shared.baseline_utils import Conv2dWithConstraint, TemporalConv2dSame


class EEGNetBackbone(nn.Module):
    """EEGNet-v4 style feature extractor for raw EEG."""

    def __init__(
        self,
        num_channels,
        n_times,
        F1=8,
        D=2,
        kernel_length=64,
        separable_kernel_length=16,
        dropout=0.25,
    ):
        super().__init__()
        F2 = F1 * D

        self.block1 = nn.Sequential(
            TemporalConv2dSame(1, F1, kernel_size=(1, kernel_length), bias=False),
            nn.BatchNorm2d(F1, eps=1e-3, momentum=0.01),
            Conv2dWithConstraint(
                F1,
                F1 * D,
                kernel_size=(num_channels, 1),
                groups=F1,
                bias=False,
                max_norm=1.0,
            ),
            nn.BatchNorm2d(F1 * D, eps=1e-3, momentum=0.01),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            TemporalConv2dSame(
                F1 * D,
                F1 * D,
                kernel_size=(1, separable_kernel_length),
                groups=F1 * D,
                bias=False,
            ),
            nn.Conv2d(F1 * D, F2, kernel_size=1, bias=False),
            nn.BatchNorm2d(F2, eps=1e-3, momentum=0.01),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
            nn.Dropout(dropout),
        )
        self.feature_dim = self._infer_feature_dim(num_channels=num_channels, n_times=n_times)

    def _infer_feature_dim(self, num_channels, n_times):
        with torch.no_grad():
            dummy = torch.ones(1, 1, num_channels, n_times)
            features = self.forward(dummy)
        return int(features.flatten(start_dim=1).shape[1])

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x
