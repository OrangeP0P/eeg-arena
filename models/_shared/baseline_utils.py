"""Shared layers and helpers for scratch baseline integrations."""

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1dWithConstraint(nn.Conv1d):
    """Conv1d layer with optional max-norm projection on the weights."""

    def __init__(self, *args, max_norm: Optional[float] = None, **kwargs):
        self.max_norm = max_norm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        if self.max_norm is not None:
            with torch.no_grad():
                self.weight.renorm_(p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


class Conv2dWithConstraint(nn.Conv2d):
    """Conv2d layer with optional max-norm projection on the weights."""

    def __init__(self, *args, max_norm: Optional[float] = None, **kwargs):
        self.max_norm = max_norm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        if self.max_norm is not None:
            with torch.no_grad():
                self.weight.renorm_(p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


class LinearWithConstraint(nn.Linear):
    """Linear layer with optional max-norm projection on the weights."""

    def __init__(self, *args, max_norm: Optional[float] = None, **kwargs):
        self.max_norm = max_norm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        if self.max_norm is not None:
            with torch.no_grad():
                self.weight.renorm_(p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


class LogVarLayer(nn.Module):
    """Log-variance pooling used by filter-bank style models."""

    def forward(self, x):
        return torch.log(torch.clamp(x.var(dim=-1), min=1e-6, max=1e6))


class LogPowerLayer(nn.Module):
    """Log-power pooling used by IFNet-like models."""

    def forward(self, x):
        return torch.log(torch.clamp(x.pow(2).mean(dim=-1), min=1e-4, max=1e4))


class TemporalConv2dSame(nn.Module):
    """Conv2d with explicit same padding on the temporal axis."""

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=0,
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


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional embedding with interpolation fallback."""

    def __init__(self, emb_size: int, length: int, dropout: float = 0.1):
        super().__init__()
        self.encoding = nn.Parameter(torch.randn(1, max(1, length), emb_size))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        seq_len = x.shape[1]
        encoding = self.encoding
        if encoding.shape[1] != seq_len:
            encoding = F.interpolate(
                encoding.transpose(1, 2),
                size=seq_len,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        return self.dropout(x + encoding.to(device=x.device, dtype=x.dtype))


class MultiHeadAttention(nn.Module):
    """Lightweight self-attention block with batch-first tensors."""

    def __init__(self, emb_size: int, num_heads: int, dropout: float):
        super().__init__()
        if emb_size % num_heads != 0:
            raise ValueError(
                f"Embedding size {emb_size} must be divisible by num_heads {num_heads}."
            )
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.proj = nn.Linear(emb_size, emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        q = self.queries(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.keys(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.values(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.emb_size)
        return self.proj(out)


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int, drop_p: float):
        super().__init__(
            nn.Linear(emb_size, emb_size * expansion),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(emb_size * expansion, emb_size),
        )


class ResidualAdd(nn.Module):
    def __init__(self, fn: nn.Module, emb_size: int, drop_p: float):
        super().__init__()
        self.fn = fn
        self.dropout = nn.Dropout(drop_p)
        self.layernorm = nn.LayerNorm(emb_size)

    def forward(self, x):
        return self.layernorm(x + self.dropout(self.fn(x)))


class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_size: int, num_heads: int, dropout: float, ffn_expansion: int = 4):
        super().__init__()
        self.attn = ResidualAdd(
            MultiHeadAttention(emb_size, num_heads, dropout),
            emb_size=emb_size,
            drop_p=dropout,
        )
        self.ffn = ResidualAdd(
            FeedForwardBlock(emb_size, expansion=ffn_expansion, drop_p=dropout),
            emb_size=emb_size,
            drop_p=dropout,
        )

    def forward(self, x):
        x = self.attn(x)
        x = self.ffn(x)
        return x


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int, emb_size: int, num_heads: int, dropout: float, ffn_expansion: int = 4):
        super().__init__(
            *[
                TransformerEncoderBlock(
                    emb_size=emb_size,
                    num_heads=num_heads,
                    dropout=dropout,
                    ffn_expansion=ffn_expansion,
                )
                for _ in range(depth)
            ]
        )


class AttentionPool1d(nn.Module):
    """Learned attention pooling over sequence tokens."""

    def __init__(self, emb_size: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.Tanh(),
            nn.Linear(emb_size, 1),
        )

    def forward(self, x):
        weights = torch.softmax(self.score(x), dim=1)
        return torch.sum(weights * x, dim=1)


def pad_to_multiple(x, multiple: int):
    """Right-pad the temporal axis so it becomes divisible by ``multiple``."""

    if multiple <= 0:
        return x
    remainder = x.shape[-1] % multiple
    if remainder == 0:
        return x
    return F.pad(x, (0, multiple - remainder))


def segment_statistics(x, segments: int):
    """Return per-segment summary statistics for ``[B, C, T]`` inputs."""

    mean = F.adaptive_avg_pool1d(x, segments)
    mean_sq = F.adaptive_avg_pool1d(x.pow(2), segments)
    std = torch.sqrt(torch.clamp(mean_sq - mean.pow(2), min=1e-6))
    rms = torch.sqrt(torch.clamp(mean_sq, min=1e-6))
    maxv = F.adaptive_max_pool1d(x, segments)
    minv = -F.adaptive_max_pool1d(-x, segments)
    feats = torch.stack([mean, std, rms, maxv, minv], dim=-1)
    return feats.permute(0, 2, 1, 3).flatten(2)


def infer_hemisphere_indices(
    ch_positions: Optional[Dict[str, Sequence[float]]],
    num_channels: int,
) -> Tuple[List[int], List[int]]:
    """Split channels into left/right groups using x coordinates when available."""

    if not ch_positions:
        midpoint = max(1, num_channels // 2)
        return list(range(midpoint)), list(range(midpoint, num_channels))

    coords = list(ch_positions.values())[:num_channels]
    left: List[int] = []
    right: List[int] = []
    center: List[int] = []
    for idx, coord in enumerate(coords):
        x = float(coord[0]) if coord else 0.0
        if x < -1e-4:
            left.append(idx)
        elif x > 1e-4:
            right.append(idx)
        else:
            center.append(idx)

    for offset, idx in enumerate(center):
        if offset % 2 == 0:
            left.append(idx)
        else:
            right.append(idx)

    if not left or not right:
        midpoint = max(1, num_channels // 2)
        return list(range(midpoint)), list(range(midpoint, num_channels))
    return sorted(left), sorted(right)


def build_region_groups(
    ch_positions: Optional[Dict[str, Sequence[float]]],
    num_channels: int,
    num_groups: int = 9,
) -> List[List[int]]:
    """Build 9 coarse spatial groups for HSLT-style region transformers."""

    if not ch_positions:
        step = max(1, math.ceil(num_channels / float(num_groups)))
        groups = [list(range(i, min(i + step, num_channels))) for i in range(0, num_channels, step)]
        while len(groups) < num_groups:
            groups.append(groups[-1][:] if groups else [0])
        return groups[:num_groups]

    coords = []
    for idx, coord in enumerate(list(ch_positions.values())[:num_channels]):
        x = float(coord[0]) if coord else 0.0
        y = float(coord[1]) if coord else 0.0
        coords.append((idx, x, y))

    ys = sorted(y for _, _, y in coords)
    lower = ys[len(ys) // 3]
    upper = ys[(2 * len(ys)) // 3]

    groups = [[] for _ in range(num_groups)]
    for idx, x, y in coords:
        if y >= upper:
            band = 0
        elif y >= lower:
            band = 1
        else:
            band = 2

        if x < -1e-4:
            column = 0
        elif x > 1e-4:
            column = 2
        else:
            column = 1
        groups[band * 3 + column].append(idx)

    if any(len(group) == 0 for group in groups):
        step = max(1, math.ceil(num_channels / float(num_groups)))
        groups = [list(range(i, min(i + step, num_channels))) for i in range(0, num_channels, step)]
        while len(groups) < num_groups:
            groups.append(groups[-1][:] if groups else [0])
        return groups[:num_groups]

    return [sorted(group) for group in groups]


def relative_bandpower(x, sample_rate: float, bands: Iterable[Tuple[float, float]]):
    """Compute relative band powers for each channel of ``[B, C, T]`` EEG."""

    x = x - x.mean(dim=-1, keepdim=True)
    spectrum = torch.fft.rfft(x, dim=-1).abs().pow(2)
    freqs = torch.fft.rfftfreq(x.shape[-1], d=1.0 / float(sample_rate)).to(x.device)
    total_power = spectrum.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    features = []
    for low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        if mask.any():
            band_power = spectrum[..., mask].sum(dim=-1, keepdim=True)
        else:
            band_power = torch.zeros_like(total_power)
        features.append(band_power / total_power)
    return torch.cat(features, dim=-1)


__all__ = [
    "AttentionPool1d",
    "Conv1dWithConstraint",
    "Conv2dWithConstraint",
    "FeedForwardBlock",
    "LinearWithConstraint",
    "LearnablePositionalEncoding",
    "LogPowerLayer",
    "LogVarLayer",
    "MultiHeadAttention",
    "ResidualAdd",
    "TemporalConv2dSame",
    "TransformerEncoder",
    "TransformerEncoderBlock",
    "build_region_groups",
    "infer_hemisphere_indices",
    "pad_to_multiple",
    "relative_bandpower",
    "segment_statistics",
]
