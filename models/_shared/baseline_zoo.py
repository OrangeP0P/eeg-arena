"""Shared scratch baseline backbones used by multiple model packages."""

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .baseline_utils import (
    AttentionPool1d,
    Conv1dWithConstraint,
    Conv2dWithConstraint,
    LearnablePositionalEncoding,
    LogPowerLayer,
    TransformerEncoder,
    build_region_groups,
    infer_hemisphere_indices,
    pad_to_multiple,
    relative_bandpower,
    segment_statistics,
)


def _coerce_raw_input(x):
    if x.ndim != 3:
        raise ValueError(f"Expected raw EEG input [B, C, T], got shape {tuple(x.shape)}")
    return x


class _ConvBNAct1d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, groups=1):
        if padding is None:
            padding = kernel_size // 2
        super().__init__(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )


class _SleepBranch(nn.Module):
    def __init__(self, in_channels, first_kernel, first_stride, pool_size, conv_kernel):
        super().__init__()
        self.net = nn.Sequential(
            _ConvBNAct1d(in_channels, 64, first_kernel, stride=first_stride, padding=first_kernel // 2),
            nn.MaxPool1d(pool_size, stride=pool_size),
            nn.Dropout(0.5),
            _ConvBNAct1d(64, 128, conv_kernel),
            _ConvBNAct1d(128, 128, conv_kernel),
            _ConvBNAct1d(128, 128, conv_kernel),
        )

    def forward(self, x):
        return self.net(x)


class FBCNetBackbone(nn.Module):
    """PyTorch fine-tune adaptation of a filter-bank style spatial CNN."""

    def __init__(self, num_channels, kernels=(15, 31, 63), m=16, stat_segments=4):
        super().__init__()
        self.stat_segments = int(stat_segments)
        branches = []
        for kernel in kernels:
            branches.append(
                nn.Sequential(
                    Conv2dWithConstraint(1, m, kernel_size=(1, kernel), padding=(0, kernel // 2), bias=False, max_norm=2.0),
                    nn.BatchNorm2d(m),
                    nn.SiLU(),
                    Conv2dWithConstraint(m, m, kernel_size=(num_channels, 1), groups=m, bias=False, max_norm=2.0),
                    nn.BatchNorm2d(m),
                    nn.SiLU(),
                )
            )
        self.branches = nn.ModuleList(branches)
        self.feature_dim = len(kernels) * m * self.stat_segments

    def forward(self, x):
        x = _coerce_raw_input(x).unsqueeze(1)
        features = torch.cat([branch(x) for branch in self.branches], dim=1).squeeze(2)
        power = F.adaptive_avg_pool1d(features.pow(2), self.stat_segments)
        return torch.log(power.clamp_min(1e-6)).flatten(1)


class IFNetBackbone(nn.Module):
    """Interactive-frequency CNN adapted to raw-input fine-tuning."""

    def __init__(self, num_channels, samples, out_planes=64, radix=3, kernel_size=63, patch_size=125, dropout=0.5):
        super().__init__()
        self.out_planes = int(out_planes)
        self.radix = int(radix)
        self.patch_size = int(patch_size)
        self.patch_num = max(1, math.ceil(samples / float(self.patch_size)))
        self.stem = nn.Sequential(
            nn.Conv1d(num_channels, self.out_planes * self.radix, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.out_planes * self.radix),
        )
        kernels = []
        current_kernel = int(kernel_size)
        for _ in range(self.radix):
            kernels.append(max(3, current_kernel))
            current_kernel = max(3, current_kernel // 2)
        self.temporal_convs = nn.ModuleList(
            [
                nn.Sequential(
                    Conv1dWithConstraint(
                        self.out_planes,
                        self.out_planes,
                        kernel_size=k,
                        padding=k // 2,
                        groups=self.out_planes,
                        bias=False,
                        max_norm=0.5,
                    ),
                    nn.BatchNorm1d(self.out_planes),
                )
                for k in kernels
            ]
        )
        self.log_power = LogPowerLayer()
        self.dropout = nn.Dropout(dropout)
        self.feature_dim = self.out_planes * self.patch_num

    def forward(self, x):
        x = _coerce_raw_input(x)
        x = self.stem(x)
        chunks = torch.chunk(x, self.radix, dim=1)
        fused = 0
        for chunk, block in zip(chunks, self.temporal_convs):
            fused = fused + block(chunk)
        fused = F.gelu(fused)
        fused = pad_to_multiple(fused, self.patch_size)
        fused = fused.view(fused.shape[0], self.out_planes, -1, self.patch_size)
        power = self.log_power(fused)
        return self.dropout(power).flatten(1)


class EEGWaveNetBackbone(nn.Module):
    """Wavelet-style temporal pyramid adapted from EEGWaveNet."""

    def __init__(self, num_channels):
        super().__init__()
        self.temporal_convs = nn.ModuleList(
            [
                nn.Conv1d(num_channels, num_channels, kernel_size=2, stride=2, groups=num_channels)
                for _ in range(6)
            ]
        )
        self.channel_pools = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(num_channels, 32, kernel_size=4, padding=3),
                    nn.BatchNorm1d(32),
                    nn.LeakyReLU(0.01),
                    nn.Conv1d(32, 32, kernel_size=4, padding=3),
                    nn.BatchNorm1d(32),
                    nn.LeakyReLU(0.01),
                )
                for _ in range(5)
            ]
        )
        self.feature_dim = 160

    def forward(self, x):
        x = _coerce_raw_input(x)
        states = []
        hidden = self.temporal_convs[0](x)
        for conv in self.temporal_convs[1:]:
            hidden = conv(hidden)
            states.append(hidden)
        pooled = [block(state).mean(dim=-1) for block, state in zip(self.channel_pools, states)]
        return torch.cat(pooled, dim=1)


class _SlimSeizBlock(nn.Module):
    def __init__(self, hidden_dim, dilation, dropout):
        super().__init__()
        self.depthwise = nn.Conv1d(
            hidden_dim,
            hidden_dim * 2,
            kernel_size=5,
            padding=2 * dilation,
            dilation=dilation,
            groups=hidden_dim,
        )
        self.pointwise = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        squeeze = max(4, hidden_dim // 4)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(hidden_dim, squeeze, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(squeeze, hidden_dim, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        residual = x
        value, gate = self.depthwise(x).chunk(2, dim=1)
        x = torch.tanh(value) * torch.sigmoid(gate)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.dropout(x)
        x = x * self.channel_gate(x)
        return residual + x


class SlimSeizBackbone(nn.Module):
    """Mamba-inspired lightweight seizure baseline without external dependencies."""

    def __init__(self, num_channels, hidden_dim=96, depth=4, dropout=0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(num_channels, hidden_dim, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[_SlimSeizBlock(hidden_dim, dilation=2 ** idx, dropout=dropout) for idx in range(depth)]
        )
        self.proj = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.feature_dim = hidden_dim * 2

    def forward(self, x):
        x = _coerce_raw_input(x)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.proj(x)
        return torch.cat([x.mean(dim=-1), x.amax(dim=-1)], dim=1)


class MSCFormerBackbone(nn.Module):
    """Multi-scale convolution + transformer encoder from MSCFormer."""

    def __init__(self, num_channels, samples, f1=16, pooling_size=52, heads=8, depth=5, dropout=0.25):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, f1, kernel_size=(1, kernel), padding=(0, kernel // 2), bias=False),
                    nn.Conv2d(f1, f1, kernel_size=(num_channels, 1), groups=f1, bias=False),
                    nn.BatchNorm2d(f1),
                    nn.ELU(),
                    nn.AvgPool2d((1, pooling_size)),
                    nn.Dropout(dropout),
                )
                for kernel in (85, 65, 45)
            ]
        )
        self.emb_size = f1 * len(self.branches)
        token_len = self._infer_token_length(num_channels, samples)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.emb_size))
        self.position = LearnablePositionalEncoding(self.emb_size, token_len + 1, dropout=0.1)
        self.transformer = TransformerEncoder(depth=depth, emb_size=self.emb_size, num_heads=heads, dropout=dropout)
        self.feature_dim = self.emb_size

    def _infer_token_length(self, num_channels, samples):
        with torch.no_grad():
            dummy = torch.ones(1, 1, num_channels, samples)
            tokens = torch.cat([branch(dummy) for branch in self.branches], dim=1)
            return int(tokens.squeeze(2).shape[-1])

    def forward(self, x):
        x = _coerce_raw_input(x).unsqueeze(1)
        x = torch.cat([branch(x) for branch in self.branches], dim=1)
        x = x.squeeze(2).transpose(1, 2)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.position(x)
        x = self.transformer(x)
        return x[:, 0]


class TMSANetBackbone(nn.Module):
    """Temporal multi-scale attention network adapted to raw-input EEG."""

    def __init__(self, num_channels, samples, embed_dim=32, pool_size=50, pool_stride=15, heads=4, depth=2, dropout=0.5):
        super().__init__()
        self.extract = nn.Sequential(
            nn.Conv2d(1, embed_dim, kernel_size=(1, 31), padding=(0, 15)),
            nn.Conv2d(1, embed_dim, kernel_size=(1, 15), padding=(0, 7)),
        )
        self.bn1 = nn.BatchNorm2d(embed_dim)
        self.spatial = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=(num_channels, 1), bias=False),
            nn.GELU(),
            nn.BatchNorm2d(embed_dim),
        )
        self.pool = nn.AvgPool1d(pool_size, stride=pool_stride)
        token_len = self._infer_token_length(num_channels, samples, embed_dim)
        self.position = LearnablePositionalEncoding(embed_dim, token_len, dropout=dropout)
        self.transformer = TransformerEncoder(depth=depth, emb_size=embed_dim, num_heads=heads, dropout=dropout)
        self.pooler = AttentionPool1d(embed_dim)
        self.feature_dim = embed_dim

    def _infer_token_length(self, num_channels, samples, embed_dim):
        with torch.no_grad():
            dummy = torch.ones(1, num_channels, samples)
            hidden = self._extract_tokens(dummy)
        return int(hidden.shape[1])

    def _extract_tokens(self, x):
        x = x.unsqueeze(1)
        temporal = self.extract[0](x) + self.extract[1](x)
        temporal = self.bn1(temporal)
        temporal = self.spatial(temporal).squeeze(2)
        temporal = self.pool(temporal)
        return temporal.transpose(1, 2)

    def forward(self, x):
        x = _coerce_raw_input(x)
        x = self._extract_tokens(x)
        x = self.position(x)
        x = self.transformer(x)
        return self.pooler(x)


class ADFCNNBackbone(nn.Module):
    """Attention-based dual-scale fusion CNN."""

    def __init__(self, num_channels, dropout=0.25, f1=8, d=1):
        super().__init__()
        f2 = f1 * d
        self.spectral_1 = nn.Sequential(
            Conv2dWithConstraint(1, f1, kernel_size=(1, 125), padding=(0, 62), max_norm=2.0),
            nn.BatchNorm2d(f1),
        )
        self.spectral_2 = nn.Sequential(
            Conv2dWithConstraint(1, f1, kernel_size=(1, 31), padding=(0, 15), max_norm=2.0),
            nn.BatchNorm2d(f1),
        )
        self.spatial_1 = nn.Sequential(
            Conv2dWithConstraint(f2, f2, kernel_size=(num_channels, 1), groups=f2, bias=False, max_norm=2.0),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.Dropout(dropout),
            Conv2dWithConstraint(f2, f2, kernel_size=(1, 1), max_norm=2.0),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 32), stride=(1, 32)),
            nn.Dropout(dropout),
        )
        self.spatial_2 = nn.Sequential(
            Conv2dWithConstraint(f2, f2, kernel_size=(num_channels, 1), max_norm=2.0),
            nn.BatchNorm2d(f2),
            nn.AvgPool2d((1, 75), stride=(1, 25)),
            nn.Dropout(dropout),
        )
        self.w_q = nn.Linear(f2, f2)
        self.w_k = nn.Linear(f2, f2)
        self.w_v = nn.Linear(f2, f2)
        self.dropout = nn.Dropout(dropout)
        self.feature_dim = f2 * 2

    def forward(self, x):
        x = _coerce_raw_input(x).unsqueeze(1)
        branch_1 = self.spatial_1(self.spectral_1(x))
        branch_2 = self.spatial_2(self.spectral_2(x))
        branch_2 = torch.log(torch.clamp(branch_2.pow(2), min=1e-6))
        merged = torch.cat([branch_1, branch_2], dim=3)
        tokens = merged.flatten(2).transpose(1, 2)
        q = F.normalize(self.w_q(tokens), dim=-1)
        k = F.normalize(self.w_k(tokens), dim=-1)
        v = self.w_v(tokens)
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1]), dim=-1)
        tokens = tokens + self.dropout(torch.matmul(attn, v))
        return torch.cat([tokens.mean(dim=1), tokens.amax(dim=1)], dim=1)


class DBConformerBackbone(nn.Module):
    """Dual-branch CNN-transformer encoder from DBConformer."""

    def __init__(
        self,
        num_channels,
        samples,
        emb_size=40,
        tem_depth=4,
        spa_depth=4,
        radix=2,
        kernel_size=63,
        temporal_patch_size=125,
        spa_dim=32,
        dropout=0.5,
        use_channel_attention=True,
    ):
        super().__init__()
        self.emb_size = emb_size
        self.temporal_patch_size = int(temporal_patch_size)
        self.patch_num = max(1, math.ceil(samples / float(self.temporal_patch_size)))
        self.temporal_stem = nn.Sequential(
            nn.Conv1d(num_channels, emb_size * radix, kernel_size=1, bias=False),
            nn.BatchNorm1d(emb_size * radix),
        )
        kernels = [max(3, kernel_size // (2 ** idx)) for idx in range(radix)]
        self.temporal_branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(emb_size, emb_size, kernel_size=k, padding=k // 2, groups=emb_size, bias=False),
                    nn.BatchNorm1d(emb_size),
                )
                for k in kernels
            ]
        )
        self.temporal_dropout = nn.Dropout(dropout)

        self.spatial_conv = nn.Sequential(
            nn.Conv1d(1, spa_dim, kernel_size=25, stride=5, padding=12),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.spatial_proj = nn.Linear(spa_dim, emb_size)
        self.temporal_position = LearnablePositionalEncoding(emb_size, self.patch_num, dropout=0.1)
        self.spatial_position = LearnablePositionalEncoding(emb_size, num_channels, dropout=0.1)
        self.temporal_transformer = TransformerEncoder(tem_depth, emb_size, num_heads=4, dropout=dropout)
        self.spatial_transformer = TransformerEncoder(spa_depth, emb_size, num_heads=4, dropout=dropout)
        self.use_channel_attention = use_channel_attention
        self.channel_pool = AttentionPool1d(emb_size) if use_channel_attention else None
        self.feature_dim = emb_size * 2

    def _temporal_tokens(self, x):
        hidden = self.temporal_stem(x)
        chunks = torch.chunk(hidden, len(self.temporal_branches), dim=1)
        fused = 0
        for chunk, branch in zip(chunks, self.temporal_branches):
            fused = fused + branch(chunk)
        fused = F.gelu(fused)
        fused = self.temporal_dropout(fused)
        tokens = F.adaptive_avg_pool1d(fused, self.patch_num).transpose(1, 2)
        return tokens

    def _spatial_tokens(self, x):
        batch_size, num_channels, samples = x.shape
        tokens = x.reshape(batch_size * num_channels, 1, samples)
        tokens = self.spatial_conv(tokens).squeeze(-1)
        tokens = self.spatial_proj(tokens).view(batch_size, num_channels, self.emb_size)
        return tokens

    def forward(self, x):
        x = _coerce_raw_input(x)
        temporal_tokens = self.temporal_position(self._temporal_tokens(x))
        spatial_tokens = self.spatial_position(self._spatial_tokens(x))
        temporal_tokens = self.temporal_transformer(temporal_tokens)
        spatial_tokens = self.spatial_transformer(spatial_tokens)
        temporal_feature = temporal_tokens.mean(dim=1)
        if self.channel_pool is not None:
            spatial_feature = self.channel_pool(spatial_tokens)
        else:
            spatial_feature = spatial_tokens.mean(dim=1)
        return torch.cat([temporal_feature, spatial_feature], dim=1)


class TSceptionBackbone(nn.Module):
    """TSception backbone for emotion-recognition style EEG classification."""

    def __init__(self, num_channels, sampling_rate=200, num_T=15, num_S=15):
        super().__init__()
        self.num_S = int(num_S)
        windows = [0.5, 0.25, 0.125]

        def conv_block(in_chan, out_chan, kernel, step, pool):
            return nn.Sequential(
                nn.Conv2d(in_chan, out_chan, kernel_size=kernel, stride=step),
                nn.LeakyReLU(),
                nn.AvgPool2d(kernel_size=(1, pool), stride=(1, pool)),
            )

        self.Tception1 = conv_block(1, num_T, (1, int(windows[0] * sampling_rate)), 1, 8)
        self.Tception2 = conv_block(1, num_T, (1, int(windows[1] * sampling_rate)), 1, 8)
        self.Tception3 = conv_block(1, num_T, (1, int(windows[2] * sampling_rate)), 1, 8)
        self.Sception1 = conv_block(num_T, num_S, (num_channels, 1), 1, 2)
        self.Sception2 = conv_block(num_T, num_S, (max(1, num_channels // 2), 1), (max(1, num_channels // 2), 1), 2)
        self.fusion = conv_block(num_S, num_S, (3, 1), 1, 4)
        self.bn_t = nn.BatchNorm2d(num_T)
        self.bn_s = nn.BatchNorm2d(num_S)
        self.bn_f = nn.BatchNorm2d(num_S)
        self.feature_dim = self.num_S

    def forward(self, x):
        x = _coerce_raw_input(x).unsqueeze(1)
        temporal = torch.cat([self.Tception1(x), self.Tception2(x), self.Tception3(x)], dim=-1)
        temporal = self.bn_t(temporal)
        spatial = torch.cat([self.Sception1(temporal), self.Sception2(temporal)], dim=2)
        spatial = self.bn_s(spatial)
        fused = self.bn_f(self.fusion(spatial))
        return torch.squeeze(torch.mean(fused, dim=-1), dim=-1)


class _HSLTRegionEncoder(nn.Module):
    def __init__(self, input_dim, emb_size, region_tokens=4, depth=2, heads=4, dropout=0.1):
        super().__init__()
        self.region_tokens = max(2, region_tokens)
        self.input_proj = nn.Linear(input_dim, emb_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.position = LearnablePositionalEncoding(emb_size, self.region_tokens + 4, dropout=dropout)
        self.transformer = TransformerEncoder(depth=depth, emb_size=emb_size, num_heads=heads, dropout=dropout)

    def forward(self, x):
        tokens = self.input_proj(x)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.position(tokens)
        tokens = self.transformer(tokens)
        cls = tokens[:, :1]
        body = tokens[:, 1:].transpose(1, 2)
        body = F.adaptive_avg_pool1d(body, self.region_tokens - 1).transpose(1, 2)
        return torch.cat([cls, body], dim=1)


class HSLTBackbone(nn.Module):
    """Hierarchical spatial transformer adapted to generic EEG montages."""

    def __init__(self, num_channels, ch_positions=None, feature_bins=5, De=8, Dr=16, Le=2, Lr=2):
        super().__init__()
        self.groups = build_region_groups(ch_positions, num_channels, num_groups=9)
        self.feature_bins = int(feature_bins)
        self.region_encoder = _HSLTRegionEncoder(feature_bins, De, region_tokens=4, depth=Le, heads=4, dropout=0.1)
        self.region_proj = nn.Linear(4 * De, Dr)
        self.region_cls = nn.Parameter(torch.randn(1, 1, Dr))
        self.region_position = LearnablePositionalEncoding(Dr, len(self.groups) + 1, dropout=0.1)
        self.region_transformer = TransformerEncoder(depth=Lr, emb_size=Dr, num_heads=4, dropout=0.1)
        self.feature_dim = Dr

    def forward(self, x):
        x = _coerce_raw_input(x)
        channel_features = torch.log1p(F.adaptive_avg_pool1d(x.abs(), self.feature_bins))
        region_features = []
        for indices in self.groups:
            region = channel_features[:, indices, :]
            encoded = self.region_encoder(region).flatten(1)
            region_features.append(self.region_proj(encoded))
        regions = torch.stack(region_features, dim=1)
        cls = self.region_cls.expand(regions.shape[0], -1, -1)
        regions = torch.cat([cls, regions], dim=1)
        regions = self.region_position(regions)
        regions = self.region_transformer(regions)
        return regions[:, 0]


class BiDANNBackbone(nn.Module):
    """Bi-hemisphere sequence encoder without the domain-adversarial heads."""

    def __init__(self, num_channels, ch_positions=None, segments=9, hidden_size=128, projected_steps=8, dropout=0.3):
        super().__init__()
        left_indices, right_indices = infer_hemisphere_indices(ch_positions, num_channels)
        self.register_buffer("left_indices", torch.tensor(left_indices, dtype=torch.long), persistent=False)
        self.register_buffer("right_indices", torch.tensor(right_indices, dtype=torch.long), persistent=False)
        self.segments = int(segments)
        self.projected_steps = int(projected_steps)
        left_input_dim = max(1, len(left_indices)) * 5
        right_input_dim = max(1, len(right_indices)) * 5
        self.left_encoder = nn.LSTM(
            input_size=left_input_dim,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.right_encoder = nn.LSTM(
            input_size=right_input_dim,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.left_norm = nn.BatchNorm1d(hidden_size)
        self.right_norm = nn.BatchNorm1d(hidden_size)
        self.feature_dim = hidden_size * self.projected_steps * 2

    def _encode(self, x, indices, lstm, bn):
        seq = segment_statistics(x.index_select(1, indices), self.segments)
        seq, _ = lstm(seq)
        seq = F.adaptive_avg_pool1d(seq.transpose(1, 2), self.projected_steps)
        seq = bn(seq)
        return F.relu(seq).flatten(1)

    def forward(self, x):
        x = _coerce_raw_input(x)
        left = self._encode(x, self.left_indices, self.left_encoder, self.left_norm)
        right = self._encode(x, self.right_indices, self.right_encoder, self.right_norm)
        return torch.cat([left, right], dim=1)


class _DualScaleSleepFrontend(nn.Module):
    def __init__(self, in_channels, target_steps=8):
        super().__init__()
        self.target_steps = int(target_steps)
        self.small = _SleepBranch(in_channels, first_kernel=50, first_stride=6, pool_size=8, conv_kernel=8)
        self.large = _SleepBranch(in_channels, first_kernel=400, first_stride=50, pool_size=4, conv_kernel=6)

    def forward(self, x):
        small = F.adaptive_avg_pool1d(self.small(x), self.target_steps)
        large = F.adaptive_avg_pool1d(self.large(x), self.target_steps)
        return torch.cat([small, large], dim=1)


class DeepSleepNetBackbone(nn.Module):
    """DeepSleepNet-inspired dual-branch CNN with intra-epoch recurrent context."""

    def __init__(self, num_channels, target_steps=8, hidden_size=128, dropout=0.5):
        super().__init__()
        self.frontend = _DualScaleSleepFrontend(num_channels, target_steps=target_steps)
        self.gru = nn.GRU(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.feature_dim = hidden_size * 4

    def forward(self, x):
        x = _coerce_raw_input(x)
        x = self.frontend(x).transpose(1, 2)
        x, _ = self.gru(x)
        return torch.cat([x.mean(dim=1), x.amax(dim=1)], dim=1)


class SleepEEGNetBackbone(nn.Module):
    """SleepEEGNet-inspired temporal encoder with attention pooling."""

    def __init__(self, num_channels, target_steps=10, hidden_size=128, dropout=0.5):
        super().__init__()
        self.frontend = nn.Sequential(
            _ConvBNAct1d(num_channels, 64, kernel_size=7),
            nn.MaxPool1d(2),
            _ConvBNAct1d(64, 128, kernel_size=5),
            nn.MaxPool1d(2),
            _ConvBNAct1d(128, 128, kernel_size=5),
        )
        self.target_steps = int(target_steps)
        self.encoder = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.pooler = AttentionPool1d(hidden_size * 2)
        self.feature_dim = hidden_size * 2

    def forward(self, x):
        x = _coerce_raw_input(x)
        x = self.frontend(x)
        x = F.adaptive_avg_pool1d(x, self.target_steps).transpose(1, 2)
        x, _ = self.encoder(x)
        return self.pooler(x)


class _UNetBlock1d(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv1d(in_channels, out_channels, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ELU(),
        )


class USleepBackbone(nn.Module):
    """Compact U-Sleep-style 1D U-Net for epoch classification."""

    def __init__(self, num_channels, base_filters=16, depth=4):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        in_channels = num_channels
        filters = base_filters
        for _ in range(depth):
            self.encoders.append(_UNetBlock1d(in_channels, filters))
            self.pools.append(nn.MaxPool1d(2))
            in_channels = filters
            filters *= 2
        self.bottom = _UNetBlock1d(in_channels, filters)
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        current = filters
        for _ in range(depth):
            next_filters = current // 2
            self.upconvs.append(nn.ConvTranspose1d(current, next_filters, kernel_size=2, stride=2))
            self.decoders.append(_UNetBlock1d(next_filters * 2, next_filters))
            current = next_filters
        self.feature_dim = base_filters * 2

    def forward(self, x):
        x = _coerce_raw_input(x)
        skips = []
        hidden = x
        for encoder, pool in zip(self.encoders, self.pools):
            hidden = encoder(hidden)
            skips.append(hidden)
            hidden = pool(hidden)
        hidden = self.bottom(hidden)
        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            hidden = upconv(hidden)
            if hidden.shape[-1] != skip.shape[-1]:
                hidden = F.interpolate(hidden, size=skip.shape[-1], mode="linear", align_corners=False)
            hidden = decoder(torch.cat([hidden, skip], dim=1))
        return torch.cat([hidden.mean(dim=-1), hidden.amax(dim=-1)], dim=1)


class YASABackbone(nn.Module):
    """Handcrafted sleep-statistics baseline inspired by YASA."""

    def __init__(self, hidden_dim=128, sample_rate=200, dropout=0.2):
        super().__init__()
        self.sample_rate = float(sample_rate)
        self.bands = (
            (0.5, 4.0),
            (4.0, 8.0),
            (8.0, 12.0),
            (12.0, 16.0),
            (16.0, 30.0),
            (30.0, 45.0),
        )
        self.project = nn.Sequential(
            nn.Linear((6 + len(self.bands)) * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.feature_dim = hidden_dim

    def forward(self, x):
        x = _coerce_raw_input(x)
        centered = x - x.mean(dim=-1, keepdim=True)
        std = centered.std(dim=-1)
        rms = torch.sqrt(torch.clamp(centered.pow(2).mean(dim=-1), min=1e-6))
        line_length = centered.diff(dim=-1).abs().mean(dim=-1)
        zero_cross = (centered[..., 1:] * centered[..., :-1] < 0).float().mean(dim=-1)
        peak = centered.abs().amax(dim=-1)
        kurt = torch.mean(centered.pow(4), dim=-1) / (std.pow(4).clamp_min(1e-6))
        stats = torch.stack([std, rms, line_length, zero_cross, peak, kurt], dim=-1)
        bands = relative_bandpower(centered, self.sample_rate, self.bands)
        features = torch.cat([stats, bands], dim=-1)
        summary = torch.cat([features.mean(dim=1), features.std(dim=1)], dim=-1)
        return self.project(summary)


class SleepTransformerBackbone(nn.Module):
    """Two-stage intra-epoch transformer adapted from SleepTransformer."""

    def __init__(
        self,
        num_channels,
        frame_size=128,
        frame_step=64,
        frame_d_model=128,
        frame_heads=4,
        frame_depth=2,
        seq_d_model=128,
        seq_heads=4,
        seq_depth=2,
        epoch_seq_len=4,
        dropout=0.3,
    ):
        super().__init__()
        self.frame_size = int(frame_size)
        self.frame_step = int(frame_step)
        self.epoch_seq_len = int(epoch_seq_len)
        self.frame_proj = nn.Sequential(
            nn.LayerNorm(num_channels * self.frame_size),
            nn.Linear(num_channels * self.frame_size, frame_d_model),
        )
        self.frame_position = LearnablePositionalEncoding(frame_d_model, 64, dropout=dropout)
        self.frame_transformer = TransformerEncoder(frame_depth, frame_d_model, frame_heads, dropout=dropout)
        self.seq_proj = nn.Linear(frame_d_model, seq_d_model)
        self.seq_position = LearnablePositionalEncoding(seq_d_model, self.epoch_seq_len, dropout=dropout)
        self.seq_transformer = TransformerEncoder(seq_depth, seq_d_model, seq_heads, dropout=dropout)
        self.pooler = AttentionPool1d(seq_d_model)
        self.feature_dim = seq_d_model

    def forward(self, x):
        x = _coerce_raw_input(x)
        if x.shape[-1] < self.frame_size:
            x = F.pad(x, (0, self.frame_size - x.shape[-1]))
        frame_count = max(1, 1 + (x.shape[-1] - self.frame_size) // self.frame_step)
        total_needed = self.frame_size + self.frame_step * (frame_count - 1)
        if total_needed > x.shape[-1]:
            x = F.pad(x, (0, total_needed - x.shape[-1]))
        frames = x.unfold(-1, self.frame_size, self.frame_step)
        frames = frames.permute(0, 2, 1, 3).reshape(x.shape[0], frame_count, -1)
        frame_tokens = self.frame_proj(frames)
        frame_tokens = self.frame_position(frame_tokens)
        frame_tokens = self.frame_transformer(frame_tokens)
        coarse = F.adaptive_avg_pool1d(frame_tokens.transpose(1, 2), self.epoch_seq_len).transpose(1, 2)
        coarse = self.seq_proj(coarse)
        coarse = self.seq_position(coarse)
        coarse = self.seq_transformer(coarse)
        return self.pooler(coarse)


__all__ = [
    "ADFCNNBackbone",
    "BiDANNBackbone",
    "DBConformerBackbone",
    "DeepSleepNetBackbone",
    "EEGWaveNetBackbone",
    "FBCNetBackbone",
    "HSLTBackbone",
    "IFNetBackbone",
    "MSCFormerBackbone",
    "SleepEEGNetBackbone",
    "SleepTransformerBackbone",
    "SlimSeizBackbone",
    "TSceptionBackbone",
    "TMSANetBackbone",
    "USleepBackbone",
    "YASABackbone",
]
