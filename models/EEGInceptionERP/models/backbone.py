import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_activation(name):
    normalized = name.lower()
    if normalized == "elu":
        return nn.ELU()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


class Conv2dWithConstraint(nn.Conv2d):
    """Conv2d layer with max-norm projection on the weights."""

    def __init__(self, *args, max_norm=None, **kwargs):
        self.max_norm = max_norm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        if self.max_norm is not None:
            with torch.no_grad():
                self.weight.renorm_(p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


class TemporalConv2dSame(nn.Module):
    """Conv2d with explicit Keras-style same padding on the temporal axis."""

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=0,
            **kwargs,
        )
        self.kernel_time = self.conv.kernel_size[0]
        self.dilation_time = self.conv.dilation[0]

    def forward(self, x):
        total_pad = self.dilation_time * (self.kernel_time - 1)
        pad_top = total_pad // 2
        pad_bottom = total_pad - pad_top
        if total_pad > 0:
            x = F.pad(x, (0, 0, pad_top, pad_bottom))
        return self.conv(x)


class EEGInceptionERPBackbone(nn.Module):
    """PyTorch translation of the official ERP-oriented EEG-Inception architecture."""

    def __init__(
        self,
        num_channels,
        n_times,
        filters_per_branch=8,
        depth_multiplier=2,
        kernel_lengths=(64, 32, 16),
        dropout=0.25,
        activation="elu",
    ):
        super().__init__()

        if n_times < 32:
            raise ValueError(
                f"EEGInceptionERP requires at least 32 time samples after preprocessing, got {n_times}"
            )

        self.kernel_lengths = tuple(int(length) for length in kernel_lengths)
        if any(length <= 0 for length in self.kernel_lengths):
            raise ValueError(f"All kernel lengths must be positive, got {self.kernel_lengths}")

        self.block1 = nn.ModuleList(
            [
                self._make_block1_branch(
                    kernel_length=kernel_length,
                    num_channels=num_channels,
                    filters_per_branch=filters_per_branch,
                    depth_multiplier=depth_multiplier,
                    dropout=dropout,
                    activation=activation,
                )
                for kernel_length in self.kernel_lengths
            ]
        )
        self.pool1 = nn.AvgPool2d(kernel_size=(4, 1), stride=(4, 1))

        block1_out_channels = filters_per_branch * len(self.kernel_lengths) * depth_multiplier
        self.block2 = nn.ModuleList(
            [
                self._make_block2_branch(
                    in_channels=block1_out_channels,
                    kernel_length=max(1, kernel_length // 4),
                    filters_per_branch=filters_per_branch,
                    dropout=dropout,
                    activation=activation,
                )
                for kernel_length in self.kernel_lengths
            ]
        )
        self.pool2 = nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1))

        block2_out_channels = filters_per_branch * len(self.kernel_lengths)
        block3_mid_channels = max(1, filters_per_branch * len(self.kernel_lengths) // 2)
        block3_out_channels = max(1, filters_per_branch * len(self.kernel_lengths) // 4)

        self.block3_stage1 = nn.Sequential(
            TemporalConv2dSame(
                block2_out_channels,
                block3_mid_channels,
                kernel_size=(8, 1),
                bias=False,
            ),
            nn.BatchNorm2d(block3_mid_channels, eps=1e-3, momentum=0.01),
            _make_activation(activation),
            nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Dropout(dropout),
        )
        self.block3_stage2 = nn.Sequential(
            TemporalConv2dSame(
                block3_mid_channels,
                block3_out_channels,
                kernel_size=(4, 1),
                bias=False,
            ),
            nn.BatchNorm2d(block3_out_channels, eps=1e-3, momentum=0.01),
            _make_activation(activation),
            nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Dropout(dropout),
        )

        self.feature_dim = self._infer_feature_dim(num_channels=num_channels, n_times=n_times)

    def _make_block1_branch(
        self,
        *,
        kernel_length,
        num_channels,
        filters_per_branch,
        depth_multiplier,
        dropout,
        activation,
    ):
        return nn.Sequential(
            TemporalConv2dSame(
                1,
                filters_per_branch,
                kernel_size=(kernel_length, 1),
                bias=True,
            ),
            nn.BatchNorm2d(filters_per_branch, eps=1e-3, momentum=0.01),
            _make_activation(activation),
            nn.Dropout(dropout),
            Conv2dWithConstraint(
                filters_per_branch,
                filters_per_branch * depth_multiplier,
                kernel_size=(1, num_channels),
                groups=filters_per_branch,
                bias=False,
                max_norm=1.0,
            ),
            nn.BatchNorm2d(filters_per_branch * depth_multiplier, eps=1e-3, momentum=0.01),
            _make_activation(activation),
            nn.Dropout(dropout),
        )

    def _make_block2_branch(
        self,
        *,
        in_channels,
        kernel_length,
        filters_per_branch,
        dropout,
        activation,
    ):
        return nn.Sequential(
            TemporalConv2dSame(
                in_channels,
                filters_per_branch,
                kernel_size=(kernel_length, 1),
                bias=False,
            ),
            nn.BatchNorm2d(filters_per_branch, eps=1e-3, momentum=0.01),
            _make_activation(activation),
            nn.Dropout(dropout),
        )

    def _infer_feature_dim(self, num_channels, n_times):
        with torch.no_grad():
            dummy = torch.ones(1, 1, n_times, num_channels)
            features = self.forward(dummy)
        return int(features.flatten(start_dim=1).shape[1])

    def forward(self, x):
        block1_outputs = [branch(x) for branch in self.block1]
        x = torch.cat(block1_outputs, dim=1)
        x = self.pool1(x)

        block2_outputs = [branch(x) for branch in self.block2]
        x = torch.cat(block2_outputs, dim=1)
        x = self.pool2(x)

        x = self.block3_stage1(x)
        x = self.block3_stage2(x)
        return x
