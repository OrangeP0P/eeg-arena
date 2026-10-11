"""RoPE transformer encoder block used by LUNA."""
import torch
import torch.nn as nn
from einops import rearrange
from timm.layers import DropPath


class RotaryEmbedding(nn.Module):
    """Minimal rotary embedding with loadable `freqs` buffer."""

    def __init__(self, dim, base=10000.0):
        super().__init__()
        if dim <= 0:
            raise ValueError("RotaryEmbedding expects dim > 0.")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("freqs", inv_freq, persistent=True)

    def _cos_sin(self, seq_len, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=self.freqs.dtype)
        freqs = torch.outer(t, self.freqs.to(device=device))
        cos = torch.cos(freqs).to(dtype=dtype)
        sin = torch.sin(freqs).to(dtype=dtype)
        return cos, sin

    def rotate_queries_or_keys(self, x):
        # x: [B, H, N, D]
        dim = x.shape[-1]
        half = dim // 2
        if half == 0:
            return x

        cos, sin = self._cos_sin(x.shape[-2], x.device, x.dtype)
        cos = cos[:, :half].unsqueeze(0).unsqueeze(0)
        sin = sin[:, :half].unsqueeze(0).unsqueeze(0)

        x1 = x[..., :half]
        x2 = x[..., half : 2 * half]
        rotated = torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)

        if dim > 2 * half:
            rotated = torch.cat((rotated, x[..., 2 * half :]), dim=-1)
        return rotated


class RotarySelfAttentionBlock(nn.Module):
    """Multi-head self attention with RoPE applied on Q/K."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        head_dim = self.dim // self.num_heads
        self.rotary_emb = RotaryEmbedding(dim=head_dim)
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv_proj = nn.Linear(self.dim, self.dim * 3, bias=qkv_bias)
        self.attn_drop_fn = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        batch, seq_len, dim = x.shape
        qkv = (
            self.qkv_proj(x)
            .reshape(batch, seq_len, 3, self.num_heads, dim // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)

        attn_weights = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_drop_fn(attn_weights)

        attn = attn_weights @ v
        attn = rearrange(attn, "B H N D -> B N (H D)")
        return self.proj_drop(self.proj(attn))


class FeedForwardBlock(nn.Module):
    """FFN block compatible with original checkpoint naming."""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class RotaryTransformerBlock(nn.Module):
    """Transformer encoder block with RoPE attention."""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = RotarySelfAttentionBlock(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = FeedForwardBlock(dim=dim, hidden_dim=int(dim * mlp_ratio), dropout=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x
