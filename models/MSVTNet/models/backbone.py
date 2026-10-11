"""Local MSVTNet backbone adapted from the upstream implementation."""

import torch
import torch.nn.functional as F
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


class MultiScaleTSConvBranch(nn.Module):
    """Single MSVTNet multi-scale temporal-spatial convolution branch."""

    def __init__(
        self,
        num_channels: int,
        temporal_filters: int,
        branch_kernel_size: int,
        temporal_kernel_size: int,
        depth_multiplier: int,
        pool_size1: int,
        pool_size2: int,
        dropout: float,
    ) -> None:
        super().__init__()
        branch_dim = temporal_filters * depth_multiplier
        self.branch_dim = branch_dim
        self.features = nn.Sequential(
            TemporalConv2dSame(
                1,
                temporal_filters,
                kernel_size=(1, branch_kernel_size),
                bias=False,
            ),
            nn.BatchNorm2d(temporal_filters),
            nn.Conv2d(
                temporal_filters,
                branch_dim,
                kernel_size=(num_channels, 1),
                groups=temporal_filters,
                bias=False,
            ),
            nn.BatchNorm2d(branch_dim),
            nn.ELU(),
            nn.AvgPool2d((1, pool_size1)),
            nn.Dropout(dropout),
            TemporalConv2dSame(
                branch_dim,
                branch_dim,
                kernel_size=(1, temporal_kernel_size),
                groups=branch_dim,
                bias=False,
            ),
            nn.BatchNorm2d(branch_dim),
            nn.ELU(),
            nn.AvgPool2d((1, pool_size2)),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        return x.squeeze(2).transpose(1, 2).contiguous()


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional embeddings with length interpolation fallback."""

    def __init__(self, seq_len: int, d_model: int) -> None:
        super().__init__()
        self.encoding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.encoding, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        encoding = self.encoding
        if x.shape[1] != encoding.shape[1]:
            encoding = F.interpolate(
                encoding.transpose(1, 2),
                size=x.shape[1],
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        return x + encoding.to(device=x.device, dtype=x.dtype)


class CrossScaleGlobalTemporalEncoder(nn.Module):
    """Transformer encoder used after concatenating branch tokens."""

    def __init__(
        self,
        seq_len: int,
        d_model: int,
        nhead: int,
        ff_ratio: float,
        dropout: float,
        num_layers: int,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(
                f"MSVTNet requires the fused feature dim ({d_model}) to be divisible by attention_heads ({nhead})."
            )

        dim_feedforward = max(d_model, int(round(d_model * ff_ratio)))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.positional_encoding = LearnablePositionalEncoding(seq_len + 1, d_model)
        self.dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        x = self.encoder(x)
        return x[:, 0]


class MSVTNetBackbone(nn.Module):
    """MSVTNet backbone that returns the global CLS feature and per-branch features."""

    def __init__(
        self,
        num_channels: int,
        n_times: int,
        branch_filters: list[int],
        branch_kernel_sizes: list[int],
        temporal_kernel_size: int = 15,
        depth_multiplier: int = 2,
        pool_size1: int = 8,
        pool_size2: int = 7,
        cnn_dropout: float = 0.3,
        attention_heads: int = 8,
        ff_ratio: float = 1.0,
        transformer_dropout: float = 0.5,
        transformer_layers: int = 2,
    ) -> None:
        super().__init__()
        if len(branch_filters) != len(branch_kernel_sizes):
            raise ValueError(
                "MSVTNet requires branch_filters and branch_kernel_sizes to have the same length."
            )
        if not branch_filters:
            raise ValueError("MSVTNet requires at least one multi-scale branch.")

        self.branches = nn.ModuleList(
            [
                MultiScaleTSConvBranch(
                    num_channels=num_channels,
                    temporal_filters=filters,
                    branch_kernel_size=kernel_size,
                    temporal_kernel_size=temporal_kernel_size,
                    depth_multiplier=depth_multiplier,
                    pool_size1=pool_size1,
                    pool_size2=pool_size2,
                    dropout=cnn_dropout,
                )
                for filters, kernel_size in zip(branch_filters, branch_kernel_sizes)
            ]
        )

        seq_len, branch_token_dims = self._infer_branch_layout(num_channels=num_channels, n_times=n_times)
        self.seq_len = seq_len
        self.branch_feature_dims = [seq_len * branch_dim for branch_dim in branch_token_dims]
        self.feature_dim = sum(branch_token_dims)
        self.transformer = CrossScaleGlobalTemporalEncoder(
            seq_len=seq_len,
            d_model=self.feature_dim,
            nhead=attention_heads,
            ff_ratio=ff_ratio,
            dropout=transformer_dropout,
            num_layers=transformer_layers,
        )

    def _infer_branch_layout(self, num_channels: int, n_times: int) -> tuple[int, list[int]]:
        with torch.no_grad():
            dummy = torch.zeros(1, 1, num_channels, n_times)
            branch_tokens = [branch(dummy) for branch in self.branches]

        seq_lens = {tokens.shape[1] for tokens in branch_tokens}
        if len(seq_lens) != 1:
            raise ValueError(
                f"MSVTNet branches produced inconsistent token lengths: {sorted(seq_lens)}"
            )
        seq_len = seq_lens.pop()
        if seq_len <= 0:
            raise ValueError(
                f"MSVTNet produced zero-length token sequences for num_channels={num_channels}, n_times={n_times}."
            )
        branch_dims = [tokens.shape[2] for tokens in branch_tokens]
        return seq_len, branch_dims

    def forward(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        branch_tokens = [branch(x) for branch in self.branches]
        branch_features = [tokens.flatten(start_dim=1) for tokens in branch_tokens]
        fused_tokens = torch.cat(branch_tokens, dim=2)
        global_feature = self.transformer(fused_tokens)
        return global_feature, branch_features
