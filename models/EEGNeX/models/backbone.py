import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from models._shared.baseline_utils import Conv2dWithConstraint, TemporalConv2dSame


class AvgPool2dSame(nn.Module):
    """AvgPool2d with Keras-style same padding."""

    def __init__(self, kernel_size, stride=None):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if stride is None:
            stride = kernel_size
        elif isinstance(stride, int):
            stride = (stride, stride)
        self.kernel_size = kernel_size
        self.stride = stride

    @staticmethod
    def _compute_pad(input_size, kernel_size, stride):
        output_size = math.ceil(input_size / stride)
        pad_total = max((output_size - 1) * stride + kernel_size - input_size, 0)
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        return pad_before, pad_after

    def forward(self, x):
        _, _, height, width = x.shape
        pad_top, pad_bottom = self._compute_pad(height, self.kernel_size[0], self.stride[0])
        pad_left, pad_right = self._compute_pad(width, self.kernel_size[1], self.stride[1])
        if pad_top or pad_bottom or pad_left or pad_right:
            x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
        return F.avg_pool2d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            count_include_pad=True,
        )


class EEGNeXBackbone(nn.Module):
    """PyTorch translation of the official EEGNeX_8_32 Keras backbone."""

    def __init__(
        self,
        num_channels,
        n_times,
        stem_filters=8,
        mid_filters=32,
        depth_multiplier=2,
        kernel_length=64,
        dilated_kernel_length=16,
        first_dilation=2,
        second_dilation=4,
        pool_size=4,
        dropout=0.5,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            TemporalConv2dSame(
                1,
                stem_filters,
                kernel_size=(1, kernel_length),
                bias=False,
            ),
            nn.BatchNorm2d(stem_filters, eps=1e-3, momentum=0.01),
            nn.ELU(),
            TemporalConv2dSame(
                stem_filters,
                mid_filters,
                kernel_size=(1, kernel_length),
                bias=False,
            ),
            nn.BatchNorm2d(mid_filters, eps=1e-3, momentum=0.01),
        )
        self.spatial_block = nn.Sequential(
            Conv2dWithConstraint(
                mid_filters,
                mid_filters * depth_multiplier,
                kernel_size=(num_channels, 1),
                groups=mid_filters,
                bias=False,
                max_norm=1.0,
            ),
            nn.BatchNorm2d(mid_filters * depth_multiplier, eps=1e-3, momentum=0.01),
            nn.ELU(),
            AvgPool2dSame(kernel_size=(1, pool_size), stride=(1, pool_size)),
            nn.Dropout(dropout),
        )
        self.dilated_block = nn.Sequential(
            TemporalConv2dSame(
                mid_filters * depth_multiplier,
                mid_filters,
                kernel_size=(1, dilated_kernel_length),
                dilation=(1, first_dilation),
                bias=False,
            ),
            nn.BatchNorm2d(mid_filters, eps=1e-3, momentum=0.01),
            TemporalConv2dSame(
                mid_filters,
                stem_filters,
                kernel_size=(1, dilated_kernel_length),
                dilation=(1, second_dilation),
                bias=False,
            ),
            nn.BatchNorm2d(stem_filters, eps=1e-3, momentum=0.01),
            nn.ELU(),
            AvgPool2dSame(kernel_size=(1, pool_size), stride=(1, pool_size)),
            nn.Dropout(dropout),
        )
        self.feature_dim = self._infer_feature_dim(num_channels=num_channels, n_times=n_times)

    def _infer_feature_dim(self, num_channels, n_times):
        with torch.no_grad():
            dummy = torch.ones(1, 1, num_channels, n_times)
            features = self.forward(dummy)
        return int(features.flatten(start_dim=1).shape[1])

    def forward(self, x):
        x = self.stem(x)
        x = self.spatial_block(x)
        x = self.dilated_block(x)
        return x
