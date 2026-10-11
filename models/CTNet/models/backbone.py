"""Local CTNet backbone adapted from the upstream project."""

import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn


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

    def forward(self, x: Tensor) -> Tensor:
        total_pad = self.dilation_time * (self.kernel_time - 1)
        left = total_pad // 2
        right = total_pad - left
        if total_pad > 0:
            x = F.pad(x, (left, right, 0, 0))
        return self.conv(x)


class PatchEmbeddingCNN(nn.Module):
    """EEGNet-style CNN frontend that produces transformer tokens."""

    def __init__(
        self,
        num_channels: int,
        eeg_f1: int = 20,
        eeg_d: int = 2,
        eeg_kernel_size: int = 64,
        eeg_temporal_kernel_size: int = 16,
        eeg_pooling_size1: int = 8,
        eeg_pooling_size2: int = 8,
        eeg_dropout: float = 0.3,
        emb_size: int = 40,
    ) -> None:
        super().__init__()
        cnn_feature_dim = eeg_f1 * eeg_d
        target_emb_size = cnn_feature_dim if emb_size <= 0 else emb_size

        self.cnn_module = nn.Sequential(
            TemporalConv2dSame(
                1,
                eeg_f1,
                kernel_size=(1, eeg_kernel_size),
                bias=False,
            ),
            nn.BatchNorm2d(eeg_f1),
            nn.Conv2d(
                eeg_f1,
                cnn_feature_dim,
                kernel_size=(num_channels, 1),
                groups=eeg_f1,
                bias=False,
            ),
            nn.BatchNorm2d(cnn_feature_dim),
            nn.ELU(),
            nn.AvgPool2d((1, eeg_pooling_size1)),
            nn.Dropout(eeg_dropout),
            TemporalConv2dSame(
                cnn_feature_dim,
                cnn_feature_dim,
                kernel_size=(1, eeg_temporal_kernel_size),
                bias=False,
            ),
            nn.BatchNorm2d(cnn_feature_dim),
            nn.ELU(),
            nn.AvgPool2d((1, eeg_pooling_size2)),
            nn.Dropout(eeg_dropout),
        )
        self.projection = (
            nn.Identity()
            if target_emb_size == cnn_feature_dim
            else nn.Conv2d(cnn_feature_dim, target_emb_size, kernel_size=(1, 1), bias=True)
        )
        self.emb_size = target_emb_size

    def forward(self, x: Tensor) -> Tensor:
        x = self.cnn_module(x)
        x = self.projection(x)
        return rearrange(x, "b e h w -> b (h w) e")


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if emb_size % num_heads != 0:
            raise ValueError(
                f"CTNet requires emb_size ({emb_size}) to be divisible by num_heads ({num_heads})."
            )
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)

        energy = torch.einsum("bhqd, bhkd -> bhqk", queries, keys)
        if mask is not None:
            fill_value = torch.finfo(energy.dtype).min
            energy = energy.masked_fill(~mask, fill_value)

        attention = F.softmax(energy / math.sqrt(self.emb_size), dim=-1)
        attention = self.att_drop(attention)
        out = torch.einsum("bhal, bhlv -> bhav", attention, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.projection(out)


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int, drop_p: float) -> None:
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class ResidualAdd(nn.Module):
    """Residual block with dropout and post-residual LayerNorm, matching the upstream CTNet layout."""

    def __init__(self, fn: nn.Module, emb_size: int, drop_p: float) -> None:
        super().__init__()
        self.fn = fn
        self.drop = nn.Dropout(drop_p)
        self.layernorm = nn.LayerNorm(emb_size)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        residual = x
        x = self.fn(x, **kwargs)
        x = self.drop(x) + residual
        return self.layernorm(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        emb_size: int,
        num_heads: int = 4,
        drop_p: float = 0.5,
        forward_expansion: int = 4,
    ) -> None:
        super().__init__()
        self.attention = ResidualAdd(
            MultiHeadAttention(emb_size, num_heads, drop_p),
            emb_size=emb_size,
            drop_p=drop_p,
        )
        self.feed_forward = ResidualAdd(
            FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=drop_p),
            emb_size=emb_size,
            drop_p=drop_p,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.attention(x)
        x = self.feed_forward(x)
        return x


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int, emb_size: int, num_heads: int, drop_p: float, forward_expansion: int) -> None:
        super().__init__(
            *[
                TransformerEncoderBlock(
                    emb_size=emb_size,
                    num_heads=num_heads,
                    drop_p=drop_p,
                    forward_expansion=forward_expansion,
                )
                for _ in range(depth)
            ]
        )


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional embedding with length interpolation fallback."""

    def __init__(self, emb_size: int, length: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.encoding = nn.Parameter(torch.randn(1, length, emb_size))

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.shape[1]
        encoding = self.encoding
        if seq_len != encoding.shape[1]:
            encoding = F.interpolate(
                encoding.transpose(1, 2),
                size=seq_len,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        x = x + encoding.to(device=x.device, dtype=x.dtype)
        return self.dropout(x)


class CTNetBackbone(nn.Module):
    """CTNet CNN-transformer encoder that returns sequence features."""

    def __init__(
        self,
        num_channels: int,
        n_times: int,
        heads: int = 4,
        depth: int = 6,
        emb_size: int = 40,
        eeg_f1: int = 20,
        eeg_d: int = 2,
        eeg_kernel_size: int = 64,
        eeg_temporal_kernel_size: int = 16,
        eeg_pooling_size1: int = 8,
        eeg_pooling_size2: int = 8,
        eeg_dropout: float = 0.3,
        attention_dropout: float = 0.5,
        position_dropout: float = 0.1,
        ffn_expansion: int = 4,
    ) -> None:
        super().__init__()
        self.patch_embedding = PatchEmbeddingCNN(
            num_channels=num_channels,
            eeg_f1=eeg_f1,
            eeg_d=eeg_d,
            eeg_kernel_size=eeg_kernel_size,
            eeg_temporal_kernel_size=eeg_temporal_kernel_size,
            eeg_pooling_size1=eeg_pooling_size1,
            eeg_pooling_size2=eeg_pooling_size2,
            eeg_dropout=eeg_dropout,
            emb_size=emb_size,
        )

        sequence_length, actual_emb_size = self._infer_token_shape(num_channels=num_channels, n_times=n_times)
        self.sequence_length = sequence_length
        self.emb_size = actual_emb_size
        self.position = LearnablePositionalEncoding(
            emb_size=self.emb_size,
            length=self.sequence_length,
            dropout=position_dropout,
        )
        self.transformer = TransformerEncoder(
            depth=depth,
            emb_size=self.emb_size,
            num_heads=heads,
            drop_p=attention_dropout,
            forward_expansion=ffn_expansion,
        )
        self.feature_dim = self.sequence_length * self.emb_size

    def _infer_token_shape(self, num_channels: int, n_times: int) -> tuple[int, int]:
        with torch.no_grad():
            dummy = torch.ones(1, 1, num_channels, n_times)
            tokens = self.patch_embedding(dummy)
        return int(tokens.shape[1]), int(tokens.shape[2])

    @staticmethod
    def _coerce_input(x: Tensor) -> Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim != 4:
            raise ValueError(f"CTNet expects input [B, C, T] or [B, 1, C, T], got shape {tuple(x.shape)}")
        if x.shape[1] != 1:
            raise ValueError(f"CTNet expects a singleton convolution channel, got shape {tuple(x.shape)}")
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self._coerce_input(x)
        cnn_tokens = self.patch_embedding(x)
        cnn_tokens = cnn_tokens * math.sqrt(self.emb_size)
        cnn_tokens = self.position(cnn_tokens)
        transformer_tokens = self.transformer(cnn_tokens)
        return cnn_tokens + transformer_tokens


__all__ = [
    "CTNetBackbone",
    "FeedForwardBlock",
    "LearnablePositionalEncoding",
    "MultiHeadAttention",
    "PatchEmbeddingCNN",
    "ResidualAdd",
    "TransformerEncoder",
    "TransformerEncoderBlock",
]
