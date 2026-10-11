"""Encoder blocks used by LEADv2."""

import torch.nn as nn

from .activations import swiglu


class EncoderLayerV2(nn.Module):
    def __init__(self, attention, d_model, d_ff, dropout, activation="relu"):
        super().__init__()
        del activation
        self.attention = attention
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.conv1 = nn.Conv1d(d_model, 2 * d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn_t, attn_c = self.attention(x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = x + self.dropout(new_x)
        y = self.norm1(x)
        y = self.conv1(y.transpose(-1, 1))
        y = swiglu(y)
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn_t, attn_c


class EncoderV2(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attns_t = []
        attns_c = []
        for attn_layer in self.attn_layers:
            x, attn_t, attn_c = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns_t.append(attn_t)
            attns_c.append(attn_c)

        if self.norm is not None:
            x = self.norm(x)
        return x, attns_t, attns_c
