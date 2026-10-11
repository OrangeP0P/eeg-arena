"""mdJPT backbone modules adapted for the unified fine-tuning framework.

This module keeps the core upstream mdJPT idea used in pretraining:
1) patch-level temporal encoding (PatchTST-style)
2) channel projection to a unified channel space
3) multi-scale channel-temporal attention convolutions
"""
from __future__ import annotations
from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class Transpose(nn.Module):
    """Transpose helper used in normalization blocks."""

    def __init__(self, *dims, contiguous: bool = False):
        super().__init__()
        self.dims = dims
        self.contiguous = contiguous

    def forward(self, x):
        x = x.transpose(*self.dims)
        return x.contiguous() if self.contiguous else x


def get_activation_fn(activation):
    if callable(activation):
        return activation()

    act = str(activation).lower()
    if act == "relu":
        return nn.ReLU()
    if act == "gelu":
        return nn.GELU()

    raise ValueError(f"Unsupported activation: {activation}")


def positional_encoding(kind: str, learn_pe: bool, q_len: int, d_model: int):
    """Upstream-compatible positional encoding initializer."""
    if kind in {"zero", "zeros"}:
        pos = torch.empty((q_len, d_model))
        nn.init.uniform_(pos, -0.02, 0.02)
    elif kind in {"normal", "gauss"}:
        pos = torch.zeros((q_len, d_model))
        nn.init.normal_(pos, mean=0.0, std=0.1)
    elif kind == "uniform":
        pos = torch.zeros((q_len, d_model))
        nn.init.uniform_(pos, a=0.0, b=0.1)
    elif kind in {None, "none"}:
        pos = torch.empty((q_len, d_model))
        nn.init.uniform_(pos, -0.02, 0.02)
        learn_pe = False
    else:
        # Keep behavior robust if users pass unsupported choices.
        pos = torch.empty((q_len, d_model))
        nn.init.uniform_(pos, -0.02, 0.02)

    return nn.Parameter(pos, requires_grad=learn_pe)


class _ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, attn_dropout: float = 0.0):
        super().__init__()
        head_dim = d_model // n_heads
        self.scale = head_dim ** -0.5
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ):
        attn_scores = torch.matmul(q, k) * self.scale

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_scores = attn_scores.masked_fill(attn_mask, float("-inf"))
            else:
                attn_scores = attn_scores + attn_mask

        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        output = torch.matmul(attn_weights, v)
        return output, attn_weights


class _MultiheadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, attn_dropout: float = 0.0, proj_dropout: float = 0.0):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        d_k = d_model // n_heads
        d_v = d_model // n_heads

        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)

        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout)
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))

    def forward(
        self,
        q: Tensor,
        k: Optional[Tensor] = None,
        v: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ):
        bs = q.size(0)
        if k is None:
            k = q
        if v is None:
            v = q

        q_s = self.W_Q(q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(k).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3, 1)
        v_s = self.W_V(v).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)

        output, attn = self.sdp_attn(
            q_s,
            k_s,
            v_s,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
        )

        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v)
        output = self.to_out(output)
        return output, attn


class TSTEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int = 256,
        norm: str = "BatchNorm",
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()

        self.self_attn = _MultiheadAttention(
            d_model,
            n_heads,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
        )

        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

    def forward(self, src: Tensor, key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):
        src2, _ = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        src = self.norm_attn(src + self.dropout_attn(src2))

        src2 = self.ff(src)
        src = self.norm_ffn(src + self.dropout_ffn(src2))
        return src


class TSTEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int = 256,
        norm: str = "BatchNorm",
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        activation: str = "gelu",
        n_layers: int = 1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TSTEncoderLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    norm=norm,
                    attn_dropout=attn_dropout,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, src: Tensor, key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):
        out = src
        for layer in self.layers:
            out = layer(out, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return out


class TSTiEncoder(nn.Module):
    """Channel-independent PatchTST encoder (upstream-equivalent structure)."""

    def __init__(
        self,
        patch_num: int,
        patch_len: int,
        n_layers: int = 3,
        d_model: int = 128,
        n_heads: int = 16,
        d_ff: int = 256,
        norm: str = "BatchNorm",
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        act: str = "gelu",
        pe: str = "zeros",
        learn_pe: bool = True,
    ):
        super().__init__()

        self.patch_num = patch_num
        self.patch_len = patch_len
        self.W_P = nn.Linear(patch_len, d_model)
        self.W_pos = positional_encoding(pe, learn_pe, patch_num, d_model)
        self.dropout = nn.Dropout(dropout)

        self.encoder = TSTEncoder(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            norm=norm,
            attn_dropout=attn_dropout,
            dropout=dropout,
            activation=act,
            n_layers=n_layers,
        )

    def _resolve_positional_encoding(self, current_patch_num: int):
        if self.W_pos.shape[0] == current_patch_num:
            return self.W_pos

        if self.W_pos.shape[0] > current_patch_num:
            return self.W_pos[:current_patch_num]

        # If sequence is longer than init, interpolate position embeddings.
        pos = self.W_pos.transpose(0, 1).unsqueeze(0)
        pos = F.interpolate(pos, size=current_patch_num, mode="linear", align_corners=False)
        pos = pos.squeeze(0).transpose(0, 1)
        return pos

    def forward(self, x) -> Tensor:
        # x: [B, C, patch_len, patch_num]
        n_vars = x.shape[1]

        x = x.permute(0, 1, 3, 2)  # [B, C, patch_num, patch_len]
        x = self.W_P(x)  # [B, C, patch_num, d_model]

        u = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        pos = self._resolve_positional_encoding(u.shape[1])
        u = self.dropout(u + pos)

        z = self.encoder(u)
        z = torch.reshape(z, (-1, n_vars, z.shape[-2], z.shape[-1]))
        z = z.permute(0, 1, 3, 2)  # [B, C, d_model, patch_num]
        return z


class PatchTSTSingleBackbone(nn.Module):
    """PatchTST-single feature backbone used in mdJPT."""

    def __init__(
        self,
        context_window: int,
        patch_len: int,
        stride: int,
        n_layers: int = 3,
        d_model: int = 128,
        n_heads: int = 16,
        d_ff: int = 256,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        act: str = "gelu",
        pe: str = "zeros",
        learn_pe: bool = True,
    ):
        super().__init__()

        self.patch_len = int(patch_len)
        self.stride = int(stride)

        # Keep patch_num stable with the expected input length from dataloader metadata.
        patch_num = max(1, int((context_window - patch_len) / stride + 1))

        self.backbone = TSTiEncoder(
            patch_num=patch_num,
            patch_len=patch_len,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            attn_dropout=attn_dropout,
            dropout=dropout,
            act=act,
            pe=pe,
            learn_pe=learn_pe,
        )

    def forward(self, z):
        # z: [B, C, T]
        if z.dim() != 3:
            raise ValueError(f"PatchTSTSingleBackbone expects 3D input [B, C, T], got {tuple(z.shape)}")

        if z.shape[-1] < self.patch_len:
            z = F.pad(z, (0, self.patch_len - z.shape[-1]))

        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        z = z.permute(0, 1, 3, 2)  # [B, C, patch_len, patch_num]
        z = self.backbone(z)
        return z


class ChannelMLP(nn.Module):
    """Map varying dataset channel counts into mdJPT's unified channel count."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.fc = nn.Linear(c_in, c_out)

    def forward(self, x):
        # x: [B, D, C, T]
        x = x.permute(0, 1, 3, 2)
        x = F.relu(self.fc(x))
        x = x.permute(0, 1, 3, 2)
        return x


class CnnPatchTST(nn.Module):
    """mdJPT multi-scale channel-temporal attention encoder."""

    def __init__(
        self,
        n_time_filters: int,
        n_ms_filters: int,
        ms_filter_time_len: int,
        n_channs: int,
        dilation_array=(1, 3, 6, 12),
        seg_att: int = 15,
        has_att: bool = True,
        global_att: bool = False,
        attention_activation: str = "softmax",
        attention_temperature: float = 1.0,
    ):
        super().__init__()

        self.ms_filter_time_len = int(ms_filter_time_len)
        self.dilation_array = [int(v) for v in dilation_array]
        if len(self.dilation_array) < 4:
            self.dilation_array = (self.dilation_array + [self.dilation_array[-1]] * 4)[:4]

        self.seg_att = int(seg_att)
        self.has_att = bool(has_att)
        self.global_att = bool(global_att)
        self.attention_activation = str(attention_activation)
        self.attention_temperature = float(attention_temperature)

        self.msConv1 = nn.Conv2d(
            n_time_filters,
            n_time_filters * n_ms_filters,
            (n_channs, self.ms_filter_time_len),
            groups=n_time_filters,
        )
        self.msConv2 = nn.Conv2d(
            n_time_filters,
            n_time_filters * n_ms_filters,
            (n_channs, self.ms_filter_time_len),
            dilation=(1, self.dilation_array[1]),
            groups=n_time_filters,
        )
        self.msConv3 = nn.Conv2d(
            n_time_filters,
            n_time_filters * n_ms_filters,
            (n_channs, self.ms_filter_time_len),
            dilation=(1, self.dilation_array[2]),
            groups=n_time_filters,
        )
        self.msConv4 = nn.Conv2d(
            n_time_filters,
            n_time_filters * n_ms_filters,
            (n_channs, self.ms_filter_time_len),
            dilation=(1, self.dilation_array[3]),
            groups=n_time_filters,
        )

        self.output_channels = n_time_filters * n_ms_filters * 4

        self.att_conv = nn.Conv2d(self.output_channels, self.output_channels, (1, self.seg_att), groups=self.output_channels)
        self.att_pool = nn.AvgPool2d((1, self.seg_att), stride=1)
        self.att_pointConv = nn.Conv2d(self.output_channels, self.output_channels, (1, 1))

    def _apply_ms_convs(self, x):
        p = [d * (self.ms_filter_time_len - 1) for d in self.dilation_array]

        out1 = self.msConv1(F.pad(x, (p[0] // 2, p[0] - p[0] // 2), "constant", 0))
        out2 = self.msConv2(F.pad(x, (p[1] // 2, p[1] - p[1] // 2), "constant", 0))
        out3 = self.msConv3(F.pad(x, (p[2] // 2, p[2] - p[2] // 2), "constant", 0))
        out4 = self.msConv4(F.pad(x, (p[3] // 2, p[3] - p[3] // 2), "constant", 0))

        return torch.cat((out1, out2, out3, out4), dim=1)

    def _apply_attention(self, x):
        if not self.has_att:
            return F.relu(x)

        att_w = F.relu(self.att_conv(F.pad(x, (self.seg_att - 1, 0), "constant", 0)))
        if self.global_att:
            att_w = torch.mean(F.pad(att_w, (self.seg_att - 1, 0), "constant", 0), dim=-1, keepdim=True)
        else:
            att_w = self.att_pool(F.pad(att_w, (self.seg_att - 1, 0), "constant", 0))
        att_w = self.att_pointConv(att_w)

        if self.attention_activation == "relu":
            att_w = F.relu(att_w)
        else:
            att_w = F.softmax(att_w / self.attention_temperature, dim=1)

        return att_w * F.relu(x)

    def forward(self, x):
        # x: [B, D, C, T]
        if x.dim() != 4:
            raise ValueError(f"CnnPatchTST expects 4D input [B, D, C, T], got {tuple(x.shape)}")

        out = self._apply_ms_convs(x)
        out = self._apply_attention(out)
        return out


class MdJPTBackbone(nn.Module):
    """Unified mdJPT backbone returning a feature map [B, D_out, 1, T_out]."""

    def __init__(
        self,
        num_input_channels: int,
        input_time_len: int,
        patch_len: int = 32,
        patch_stride: int = 6,
        d_model: int = 32,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = 256,
        transformer_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        target_channels: int = 64,
        n_ms_filters: int = 4,
        ms_filter_time_len: int = 3,
        dilation_array=(1, 3, 6, 12),
        seg_att: int = 15,
        has_att: bool = True,
        global_att: bool = False,
    ):
        super().__init__()

        self.patch_encoder = PatchTSTSingleBackbone(
            context_window=input_time_len,
            patch_len=patch_len,
            stride=patch_stride,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            attn_dropout=attn_dropout,
            dropout=transformer_dropout,
        )

        self.channel_projector = ChannelMLP(c_in=num_input_channels, c_out=target_channels)

        self.cnn_encoder = CnnPatchTST(
            n_time_filters=d_model,
            n_ms_filters=n_ms_filters,
            ms_filter_time_len=ms_filter_time_len,
            n_channs=target_channels,
            dilation_array=dilation_array,
            seg_att=seg_att,
            has_att=has_att,
            global_att=global_att,
        )

        self.feature_dim = self.cnn_encoder.output_channels

    def forward(self, x):
        # x: [B, C, T]
        z = self.patch_encoder(x)  # [B, C, d_model, patch_num]
        z = z.permute(0, 2, 1, 3)  # [B, d_model, C, patch_num]
        z = self.channel_projector(z)  # [B, d_model, target_channels, patch_num]
        z = self.cnn_encoder(z)  # [B, feature_dim, 1, patch_num]
        return z
