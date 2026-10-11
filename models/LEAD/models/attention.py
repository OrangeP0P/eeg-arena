"""Attention blocks used by LEADv2."""

from math import sqrt

import torch
import torch.nn as nn
from einops import rearrange


class FullAttention(nn.Module):
    def __init__(
        self,
        mask_flag=True,
        factor=5,
        scale=None,
        attention_dropout=0.1,
        output_attention=False,
    ):
        super().__init__()
        del factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        del tau, delta
        _, _, _, head_dim = queries.shape
        scale = self.scale or 1.0 / sqrt(head_dim)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag and attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float("-inf"))

        attn = self.dropout(torch.softmax(scale * scores, dim=-1))
        values = torch.einsum("bhls,bshd->blhd", attn, values)
        if self.output_attention:
            return values.contiguous(), attn
        return values.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        batch_size, query_len, _ = queries.shape
        _, key_len, _ = keys.shape
        heads = self.n_heads

        queries = self.query_projection(queries).view(batch_size, query_len, heads, -1)
        keys = self.key_projection(keys).view(batch_size, key_len, heads, -1)
        values = self.value_projection(values).view(batch_size, key_len, heads, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta,
        )
        out = out.view(batch_size, query_len, -1)
        return self.out_projection(out), attn


class LEADLayerV2(nn.Module):
    """Parallel temporal-spatial attention with gated fusion."""

    def __init__(
        self,
        d_model,
        enc_in,
        n_heads,
        patch_num,
        dropout=0.1,
        output_attention=False,
    ):
        super().__init__()
        self.patch_num = patch_num
        self.d_model = d_model
        self.enc_in = enc_in

        self.temporal_attention = AttentionLayer(
            FullAttention(False, factor=1, attention_dropout=dropout, output_attention=output_attention),
            d_model,
            n_heads,
        )
        self.spatial_attention = AttentionLayer(
            FullAttention(False, factor=1, attention_dropout=dropout, output_attention=output_attention),
            d_model,
            n_heads,
        )

        self.t_norm = nn.LayerNorm(d_model)
        self.c_norm = nn.LayerNorm(d_model)
        self.fuse_proj = nn.Linear(2 * d_model, d_model)
        nn.init.xavier_uniform_(self.fuse_proj.weight)
        nn.init.zeros_(self.fuse_proj.bias)

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        batch_size, total_tokens, hidden_dim = x.shape
        num_channels = self.enc_in
        num_patches = total_tokens // num_channels
        if total_tokens != num_channels * num_patches or hidden_dim != self.d_model:
            raise ValueError(
                f"LEADv2 expects [B, C*P, D] with C={num_channels}, D={self.d_model}, "
                f"but got shape {tuple(x.shape)}."
            )

        x_t = rearrange(x, "b (c n) d -> (b c) n d", b=batch_size, c=num_channels, n=num_patches)
        x_t = self.t_norm(x_t)
        x_t_out, attn_t = self.temporal_attention(x_t, x_t, x_t, attn_mask=None, tau=tau, delta=delta)
        x_t_out = rearrange(x_t_out, "(b c) n d -> b (c n) d", b=batch_size, c=num_channels)

        x_c = rearrange(x, "b (c n) d -> (b n) c d", b=batch_size, c=num_channels, n=num_patches)
        x_c = self.c_norm(x_c)
        x_c_out, attn_c = self.spatial_attention(x_c, x_c, x_c, attn_mask=None, tau=tau, delta=delta)
        x_c_out = rearrange(x_c_out, "(b n) c d -> b (c n) d", b=batch_size, n=num_patches)

        gate = torch.sigmoid(self.fuse_proj(torch.cat([x_t_out, x_c_out], dim=-1)))
        x_next = gate * x_t_out + (1.0 - gate) * x_c_out
        return x_next, attn_t, attn_c
