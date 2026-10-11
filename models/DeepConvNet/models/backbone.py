import warnings

import torch
import torch.nn as nn

from models._shared.baseline_utils import Conv2dWithConstraint

_N_BLOCKS = 4
_MIN_TIME_AFTER_CONV = 2  # BatchNorm needs > 1 value per channel


def _simulate_time_dim(n_times, filter_time_length, pool_time_length, pool_time_stride):
    """Return the time dimension after all blocks, or -1 if it collapses."""
    t = n_times
    # block1: temporal conv → spatial conv (no time change) → pool
    t = t - filter_time_length + 1
    if t < _MIN_TIME_AFTER_CONV:
        return -1
    t = (t - pool_time_length) // pool_time_stride + 1
    if t < 1:
        return -1
    # blocks 2–4: conv → pool
    for _ in range(_N_BLOCKS - 1):
        t = t - filter_time_length + 1
        if t < _MIN_TIME_AFTER_CONV:
            return -1
        t = (t - pool_time_length) // pool_time_stride + 1
        if t < 1:
            return -1
    return t


def _adapt_filter_time_length(n_times, filter_time_length, pool_time_length, pool_time_stride):
    """
    Reduce filter_time_length until the architecture produces a valid output for
    the given n_times, or raise if no valid length >= 2 exists.

    This is computed from the convolution/pooling geometry for any sequence
    length; it does not special-case dataset-specific sample counts.
    """
    for ftl in range(filter_time_length, 1, -1):
        if _simulate_time_dim(
            n_times=n_times,
            filter_time_length=ftl,
            pool_time_length=pool_time_length,
            pool_time_stride=pool_time_stride,
        ) >= 1:
            return ftl
    raise ValueError(
        f"Cannot find a valid filter_time_length for n_times={n_times} with "
        f"pool_time_length={pool_time_length}, pool_time_stride={pool_time_stride}. "
        f"The input sequence is too short for {_N_BLOCKS} convolutional blocks."
    )


class _DeepConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        filter_time_length,
        pool_time_length,
        pool_time_stride,
        dropout,
    ):
        super().__init__()
        self.net = nn.Sequential(
            Conv2dWithConstraint(
                in_channels,
                out_channels,
                kernel_size=(1, filter_time_length),
                bias=False,
                max_norm=2.0,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, pool_time_length), stride=(1, pool_time_stride)),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class DeepConvNetBackbone(nn.Module):
    """DeepConvNet-style convolutional encoder close to the classic implementation."""

    def __init__(
        self,
        num_channels,
        n_times,
        n_filters_time=25,
        filter_time_length=10,
        pool_time_length=3,
        pool_time_stride=3,
        dropout=0.5,
    ):
        super().__init__()

        adapted_ftl = _adapt_filter_time_length(
            n_times, filter_time_length, pool_time_length, pool_time_stride
        )
        if adapted_ftl != filter_time_length:
            warnings.warn(
                f"DeepConvNet: filter_time_length reduced from {filter_time_length} to "
                f"{adapted_ftl} because n_times={n_times} is too short for the default kernel.",
                UserWarning,
                stacklevel=2,
            )
        filter_time_length = adapted_ftl

        self.block1 = nn.Sequential(
            Conv2dWithConstraint(
                1,
                n_filters_time,
                kernel_size=(1, filter_time_length),
                bias=False,
                max_norm=2.0,
            ),
            Conv2dWithConstraint(
                n_filters_time,
                n_filters_time,
                kernel_size=(num_channels, 1),
                bias=False,
                max_norm=2.0,
            ),
            nn.BatchNorm2d(n_filters_time),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, pool_time_length), stride=(1, pool_time_stride)),
            nn.Dropout(dropout),
        )
        self.block2 = _DeepConvBlock(
            n_filters_time, n_filters_time * 2, filter_time_length,
            pool_time_length, pool_time_stride, dropout
        )
        self.block3 = _DeepConvBlock(
            n_filters_time * 2, n_filters_time * 4, filter_time_length,
            pool_time_length, pool_time_stride, dropout
        )
        self.block4 = _DeepConvBlock(
            n_filters_time * 4, n_filters_time * 8, filter_time_length,
            pool_time_length, pool_time_stride, dropout
        )
        self.feature_dim = self._infer_feature_dim(num_channels=num_channels, n_times=n_times)

    def _infer_feature_dim(self, num_channels, n_times):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            dummy = torch.ones(1, 1, num_channels, n_times)
            features = self.forward(dummy)
        if was_training:
            self.train()
        return int(features.flatten(start_dim=1).shape[1])

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x
