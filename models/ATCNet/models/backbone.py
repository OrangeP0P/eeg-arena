"""PyTorch ATCNet backbone adapted from the official TensorFlow architecture."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from models._shared.baseline_utils import Conv1dWithConstraint, Conv2dWithConstraint


def _make_activation(name):
    name = str(name).lower()
    if name == "elu":
        return nn.ELU()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation '{name}'.")


class TemporalConv2dSame(nn.Module):
    """Conv2d with explicit same padding on the temporal axis."""

    def __init__(self, in_channels, out_channels, kernel_size, max_norm=None, **kwargs):
        super().__init__()
        self.conv = Conv2dWithConstraint(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=0,
            max_norm=max_norm,
            **kwargs,
        )
        self.kernel_time = self.conv.kernel_size[1]
        self.dilation_time = self.conv.dilation[1]

    def forward(self, x):
        total_pad = self.dilation_time * (self.kernel_time - 1)
        left = total_pad // 2
        right = total_pad - left
        if total_pad > 0:
            x = F.pad(x, (left, right, 0, 0))
        return self.conv(x)


class CausalConv1d(nn.Module):
    """Conv1d with left-only padding to match Keras causal convolutions."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, max_norm=None, **kwargs):
        super().__init__()
        self.left_padding = dilation * (kernel_size - 1)
        self.conv = Conv1dWithConstraint(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
            max_norm=max_norm,
            **kwargs,
        )

    def forward(self, x):
        if self.left_padding > 0:
            x = F.pad(x, (self.left_padding, 0))
        return self.conv(x)


class SequenceSqueezeExcitation(nn.Module):
    """Squeeze-and-excitation on sequence features."""

    def __init__(self, channels, ratio=8):
        super().__init__()
        hidden = channels if ratio == 0 else max(1, channels // ratio)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x):
        scale = x.mean(dim=1)
        scale = F.relu(self.fc1(scale), inplace=True)
        scale = torch.sigmoid(self.fc2(scale)).unsqueeze(1)
        return x * scale


class SequenceCBAM(nn.Module):
    """CBAM block adapted to [B, L, C] sequence features."""

    def __init__(self, channels, ratio=8, kernel_size=7):
        super().__init__()
        hidden = channels if ratio == 0 else max(1, channels // ratio)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)
        self.spatial = nn.Conv1d(
            2,
            1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )

    def forward(self, x):
        avg_pool = x.mean(dim=1)
        max_pool = x.amax(dim=1)

        avg_scale = self.fc2(F.relu(self.fc1(avg_pool), inplace=True))
        max_scale = self.fc2(F.relu(self.fc1(max_pool), inplace=True))
        channel_scale = torch.sigmoid(avg_scale + max_scale).unsqueeze(1)
        x = x * channel_scale

        avg_spatial = x.mean(dim=2, keepdim=True)
        max_spatial = x.amax(dim=2, keepdim=True)
        spatial_input = torch.cat([avg_spatial, max_spatial], dim=2).transpose(1, 2)
        spatial_scale = torch.sigmoid(self.spatial(spatial_input)).transpose(1, 2)
        return x * spatial_scale


class KerasStyleSelfAttention(nn.Module):
    """Self-attention that stays close to Keras ATCNet defaults."""

    def __init__(
        self,
        input_dim,
        key_dim=8,
        num_heads=2,
        attention_dropout=0.5,
        residual_dropout=0.3,
        local=False,
    ):
        super().__init__()
        self.local = local
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.inner_dim = key_dim * num_heads

        self.norm = nn.LayerNorm(input_dim, eps=1e-6)
        self.q_proj = nn.Linear(input_dim, self.inner_dim)
        self.k_proj = nn.Linear(input_dim, self.inner_dim)
        self.v_proj = nn.Linear(input_dim, self.inner_dim)
        self.out_proj = nn.Linear(self.inner_dim, input_dim)
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.out_dropout = nn.Dropout(residual_dropout)

        if local:
            self.tau = nn.Parameter(torch.tensor(math.sqrt(float(key_dim)), dtype=torch.float32))
        else:
            self.register_parameter("tau", None)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.key_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.key_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.key_dim).transpose(1, 2)

        if self.local:
            q = q / self.tau.clamp_min(1e-6)
        else:
            q = q / math.sqrt(self.key_dim)

        attn_scores = torch.matmul(q, k.transpose(-2, -1))
        if self.local and seq_len > 1:
            diag_mask = torch.eye(seq_len, device=x.device, dtype=torch.bool).view(1, 1, seq_len, seq_len)
            attn_scores = attn_scores.masked_fill(diag_mask, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attended = torch.matmul(attn_weights, v)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.inner_dim)
        attended = self.out_proj(attended)
        attended = self.out_dropout(attended)
        return residual + attended


class ATCNetAttentionBlock(nn.Module):
    """Optional attention stage applied before each temporal branch."""

    def __init__(
        self,
        input_dim,
        attention="mha",
        attention_heads=2,
        attention_key_dim=8,
        attention_dropout=0.5,
        attention_residual_dropout=0.3,
        attention_se_ratio=8,
    ):
        super().__init__()
        normalized = str(attention).lower() if attention is not None else "none"

        if normalized == "none":
            self.block = None
        elif normalized == "mha":
            self.block = KerasStyleSelfAttention(
                input_dim=input_dim,
                key_dim=attention_key_dim,
                num_heads=attention_heads,
                attention_dropout=attention_dropout,
                residual_dropout=attention_residual_dropout,
                local=False,
            )
        elif normalized == "mhla":
            self.block = KerasStyleSelfAttention(
                input_dim=input_dim,
                key_dim=attention_key_dim,
                num_heads=attention_heads,
                attention_dropout=attention_dropout,
                residual_dropout=attention_residual_dropout,
                local=True,
            )
        elif normalized == "se":
            self.block = SequenceSqueezeExcitation(input_dim, ratio=attention_se_ratio)
        elif normalized == "cbam":
            self.block = SequenceCBAM(input_dim, ratio=attention_se_ratio)
        else:
            raise ValueError(f"Unsupported attention block '{attention}'.")

    def forward(self, x):
        if self.block is None:
            return x
        return self.block(x)


class TemporalResidualBlock(nn.Module):
    """Residual TCN block used in each ATCNet branch."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout, activation, max_norm=0.6):
        super().__init__()
        self.conv1 = CausalConv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            bias=True,
            max_norm=max_norm,
        )
        self.bn1 = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        self.act1 = _make_activation(activation)
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = CausalConv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            bias=True,
            max_norm=max_norm,
        )
        self.bn2 = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        self.act2 = _make_activation(activation)
        self.drop2 = nn.Dropout(dropout)

        self.residual_proj = None
        if in_channels != out_channels:
            self.residual_proj = Conv1dWithConstraint(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                bias=True,
                max_norm=max_norm,
            )

        self.out_activation = _make_activation(activation)

    def forward(self, x):
        residual = x if self.residual_proj is None else self.residual_proj(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.drop2(out)

        return self.out_activation(out + residual)


class ATCTemporalConvNet(nn.Module):
    """Stack of causal residual TCN blocks."""

    def __init__(self, input_dim, depth, kernel_size, filters, dropout, activation):
        super().__init__()
        layers = []
        current_dim = input_dim
        for block_index in range(depth):
            dilation = 1 if block_index == 0 else 2 ** block_index
            layers.append(
                TemporalResidualBlock(
                    in_channels=current_dim,
                    out_channels=filters,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    activation=activation,
                )
            )
            current_dim = filters
        self.blocks = nn.ModuleList(layers)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class ATCNetConvEncoder(nn.Module):
    """Shared convolutional frontend before sliding-window branches."""

    def __init__(
        self,
        num_channels,
        eegn_F1=16,
        eegn_D=2,
        eegn_kernel_size=64,
        eegn_pool_size=7,
        eegn_dropout=0.3,
    ):
        super().__init__()
        feature_dim = eegn_F1 * eegn_D

        self.temporal = TemporalConv2dSame(
            1,
            eegn_F1,
            kernel_size=(1, eegn_kernel_size),
            bias=False,
            max_norm=0.6,
        )
        self.temporal_bn = nn.BatchNorm2d(eegn_F1, eps=1e-3, momentum=0.01)

        self.spatial = Conv2dWithConstraint(
            eegn_F1,
            feature_dim,
            kernel_size=(num_channels, 1),
            groups=eegn_F1,
            bias=False,
            max_norm=0.6,
        )
        self.spatial_bn = nn.BatchNorm2d(feature_dim, eps=1e-3, momentum=0.01)
        self.spatial_act = nn.ELU()
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8))
        self.drop1 = nn.Dropout(eegn_dropout)

        self.temporal_mix = TemporalConv2dSame(
            feature_dim,
            feature_dim,
            kernel_size=(1, 16),
            bias=False,
            max_norm=0.6,
        )
        self.temporal_mix_bn = nn.BatchNorm2d(feature_dim, eps=1e-3, momentum=0.01)
        self.temporal_mix_act = nn.ELU()
        self.pool2 = nn.AvgPool2d(kernel_size=(1, eegn_pool_size), stride=(1, eegn_pool_size))
        self.drop2 = nn.Dropout(eegn_dropout)

    def forward(self, x):
        x = self.temporal(x)
        x = self.temporal_bn(x)

        x = self.spatial(x)
        x = self.spatial_bn(x)
        x = self.spatial_act(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.temporal_mix(x)
        x = self.temporal_mix_bn(x)
        x = self.temporal_mix_act(x)
        x = self.pool2(x)
        x = self.drop2(x)

        x = x.squeeze(2).transpose(1, 2)
        return x


class ATCNetWindowBranch(nn.Module):
    """One sliding-window encoder branch."""

    def __init__(
        self,
        input_dim,
        attention="mha",
        attention_heads=2,
        attention_key_dim=8,
        attention_dropout=0.5,
        attention_residual_dropout=0.3,
        attention_se_ratio=8,
        tcn_depth=2,
        tcn_kernel_size=4,
        tcn_filters=32,
        tcn_dropout=0.3,
        tcn_activation="elu",
    ):
        super().__init__()
        self.attention = ATCNetAttentionBlock(
            input_dim=input_dim,
            attention=attention,
            attention_heads=attention_heads,
            attention_key_dim=attention_key_dim,
            attention_dropout=attention_dropout,
            attention_residual_dropout=attention_residual_dropout,
            attention_se_ratio=attention_se_ratio,
        )
        self.tcn = ATCTemporalConvNet(
            input_dim=input_dim,
            depth=tcn_depth,
            kernel_size=tcn_kernel_size,
            filters=tcn_filters,
            dropout=tcn_dropout,
            activation=tcn_activation,
        )

    def forward(self, x):
        x = self.attention(x)
        x = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        return x[:, -1, :]


class ATCNetBackbone(nn.Module):
    """Shared ATCNet encoder returning one feature vector per sliding window."""

    def __init__(
        self,
        num_channels,
        n_times,
        n_windows=5,
        attention="mha",
        attention_heads=2,
        attention_key_dim=8,
        attention_dropout=0.5,
        attention_residual_dropout=0.3,
        attention_se_ratio=8,
        eegn_F1=16,
        eegn_D=2,
        eegn_kernel_size=64,
        eegn_pool_size=7,
        eegn_dropout=0.3,
        tcn_depth=2,
        tcn_kernel_size=4,
        tcn_filters=32,
        tcn_dropout=0.3,
        tcn_activation="elu",
    ):
        super().__init__()
        if n_windows <= 0:
            raise ValueError("ATCNet requires n_windows >= 1.")

        self.conv_encoder = ATCNetConvEncoder(
            num_channels=num_channels,
            eegn_F1=eegn_F1,
            eegn_D=eegn_D,
            eegn_kernel_size=eegn_kernel_size,
            eegn_pool_size=eegn_pool_size,
            eegn_dropout=eegn_dropout,
        )

        branch_input_dim = eegn_F1 * eegn_D
        self.window_branches = nn.ModuleList(
            [
                ATCNetWindowBranch(
                    input_dim=branch_input_dim,
                    attention=attention,
                    attention_heads=attention_heads,
                    attention_key_dim=attention_key_dim,
                    attention_dropout=attention_dropout,
                    attention_residual_dropout=attention_residual_dropout,
                    attention_se_ratio=attention_se_ratio,
                    tcn_depth=tcn_depth,
                    tcn_kernel_size=tcn_kernel_size,
                    tcn_filters=tcn_filters,
                    tcn_dropout=tcn_dropout,
                    tcn_activation=tcn_activation,
                )
                for _ in range(n_windows)
            ]
        )

        self.n_windows = n_windows
        self.output_dim = tcn_filters
        self.sequence_length = self._infer_sequence_length(num_channels=num_channels, n_times=n_times)
        self.window_size = self.sequence_length - self.n_windows + 1
        if self.window_size < 1:
            raise ValueError(
                "ATCNet n_windows is too large for the pooled sequence length. "
                f"Got n_windows={n_windows}, pooled_length={self.sequence_length}, "
                f"input_samples={n_times}."
            )

    def _infer_sequence_length(self, num_channels, n_times):
        with torch.no_grad():
            dummy = torch.ones(1, 1, num_channels, n_times)
            tokens = self.conv_encoder(dummy)
        return int(tokens.shape[1])

    def forward(self, x):
        tokens = self.conv_encoder(x)
        sequence_length = tokens.shape[1]
        if sequence_length < self.n_windows:
            raise ValueError(
                "ATCNet received an EEG sequence that becomes too short after pooling. "
                f"Need at least {self.n_windows} pooled steps, got {sequence_length}."
            )

        window_features = []
        for window_index, branch in enumerate(self.window_branches):
            start = window_index
            end = sequence_length - self.n_windows + window_index + 1
            branch_tokens = tokens[:, start:end, :]
            window_features.append(branch(branch_tokens))

        return torch.stack(window_features, dim=1)
