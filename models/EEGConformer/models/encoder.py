"""Local EEG-Conformer implementation based directly on the original source."""

import warnings

import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor, nn


class PatchEmbedding(nn.Module):
    """Convolutional patch embedding from the original EEG-Conformer."""

    def __init__(
        self,
        n_filters_time: int,
        filter_time_length: int,
        n_channels: int,
        pool_time_length: int,
        pool_time_stride: int,
        drop_prob: float,
    ) -> None:
        super().__init__()
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, n_filters_time, kernel_size=(1, filter_time_length), stride=(1, 1)),
            nn.Conv2d(n_filters_time, n_filters_time, kernel_size=(n_channels, 1), stride=(1, 1)),
            nn.BatchNorm2d(n_filters_time),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool_time_length), stride=(1, pool_time_stride)),
            nn.Dropout(drop_prob),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(n_filters_time, n_filters_time, kernel_size=(1, 1), stride=(1, 1)),
            Rearrange("b d_model 1 seq -> b seq d_model"),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if emb_size % num_heads != 0:
            raise ValueError(
                f"emb_size ({emb_size}) must be divisible by num_heads ({num_heads}) for EEGConformer."
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

        scaling = self.emb_size ** 0.5
        attention = F.softmax(energy / scaling, dim=-1)
        attention = self.att_drop(attention)
        out = torch.einsum("bhal, bhlv -> bhav", attention, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.projection(out)


class ResidualAdd(nn.Module):
    def __init__(self, fn: nn.Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        residual = x
        x = self.fn(x, **kwargs)
        x = x + residual
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int, drop_p: float) -> None:
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(
        self,
        emb_size: int,
        num_heads: int,
        drop_p: float,
        forward_expansion: int = 4,
    ) -> None:
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    MultiHeadAttention(emb_size, num_heads, drop_p),
                    nn.Dropout(drop_p),
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=drop_p),
                    nn.Dropout(drop_p),
                )
            ),
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int, emb_size: int, num_heads: int, drop_p: float) -> None:
        super().__init__(
            *[TransformerEncoderBlock(emb_size, num_heads=num_heads, drop_p=drop_p) for _ in range(depth)]
        )


class EEGConformerEncoder(nn.Module):
    """Encoder portion of EEG-Conformer, implemented locally from source."""

    def __init__(
        self,
        n_chans: int,
        n_times: int,
        n_filters_time: int = 40,
        filter_time_length: int = 25,
        pool_time_length: int = 75,
        pool_time_stride: int = 15,
        drop_prob: float = 0.5,
        att_depth: int = 6,
        att_heads: int = 10,
        att_drop_prob: float = 0.5,
    ) -> None:
        super().__init__()
        if n_chans > 64:
            warnings.warn(
                "EEGConformer has only been tested on no more than 64 channels in the original implementation.",
                UserWarning,
            )

        self.n_chans = n_chans
        self.n_times = n_times
        self.n_filters_time = n_filters_time

        self.patch_embedding = PatchEmbedding(
            n_filters_time=n_filters_time,
            filter_time_length=filter_time_length,
            n_channels=n_chans,
            pool_time_length=pool_time_length,
            pool_time_stride=pool_time_stride,
            drop_prob=drop_prob,
        )
        self.transformer = TransformerEncoder(
            depth=att_depth,
            emb_size=n_filters_time,
            num_heads=att_heads,
            drop_p=att_drop_prob,
        )

        self.sequence_length = self._infer_sequence_length()
        self.feature_dim = self.sequence_length * n_filters_time

    def _infer_sequence_length(self) -> int:
        with torch.no_grad():
            dummy = torch.ones(1, 1, self.n_chans, self.n_times)
            tokens = self.patch_embedding(dummy)
        return int(tokens.shape[1])

    def _coerce_input(self, x: Tensor) -> Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim != 4:
            raise ValueError(
                f"EEGConformerEncoder expects [B, C, T] or [B, 1, C, T], got shape {tuple(x.shape)}"
            )
        if x.shape[1] != 1:
            raise ValueError(
                f"EEGConformerEncoder expects a singleton convolution channel, got shape {tuple(x.shape)}"
            )
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self._coerce_input(x)
        x = self.patch_embedding(x)
        x = self.transformer(x)
        return x


class ClassificationHead(nn.Module):
    """Flatten + MLP head used by the original EEG-Conformer code."""

    def __init__(
        self,
        feature_dim: int,
        emb_size: int,
        n_classes: int,
        hidden_dim: int = 256,
        penultimate_dim: int = 32,
        drop_prob_1: float = 0.5,
        drop_prob_2: float = 0.3,
    ) -> None:
        super().__init__()
        # The original repo defines this alternative cls head but does not use it in forward.
        self.clshead = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes),
        )
        self.hidden = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(drop_prob_1),
            nn.Linear(hidden_dim, penultimate_dim),
            nn.ELU(),
            nn.Dropout(drop_prob_2),
        )
        self.classifier = nn.Linear(penultimate_dim, n_classes)

    def flatten_tokens(self, x: Tensor) -> Tensor:
        return x.contiguous().view(x.size(0), -1)

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.flatten_tokens(x)
        return self.hidden(x)

    def forward(self, x: Tensor) -> Tensor:
        features = self.forward_features(x)
        return self.classifier(features)


__all__ = [
    "ClassificationHead",
    "EEGConformerEncoder",
    "FeedForwardBlock",
    "MultiHeadAttention",
    "PatchEmbedding",
    "ResidualAdd",
    "TransformerEncoder",
    "TransformerEncoderBlock",
]
