"""NeuroRVQ EEG backbone used by the unified fine-tuning wrapper."""
import math
from dataclasses import dataclass
from functools import partial

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F


def drop_path(x, drop_prob=0.0, training=False):
    """Drop paths (stochastic depth) per sample."""
    if drop_prob == 0.0 or not training:
        return x

    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    """Fill tensor with values drawn from a truncated normal distribution."""

    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        lower = norm_cdf((a - mean) / std)
        upper = norm_cdf((b - mean) / std)

        tensor.uniform_(2 * lower - 1, 2 * upper - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        if qk_norm is not None:
            self.q_norm = qk_norm(head_dim)
            self.k_norm = qk_norm(head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        else:
            qkv_bias = None

        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.q_norm is not None:
            q = self.q_norm(q).type_as(v)
        if self.k_norm is not None:
            k = self.k_norm(k).type_as(v)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, -1)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_norm=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        init_values=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1 = None
            self.gamma_2 = None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class MultiDimensionalTemporalConv(nn.Module):
    """Multi-scale temporal frontend used by NeuroRVQ."""

    def __init__(self, in_chans=1, out_chans=8):
        super().__init__()

        self.conv1_1 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 21), padding=(0, 10))
        self.norm1_1 = nn.GroupNorm(4, out_chans)
        self.pool1_1 = nn.AvgPool2d(kernel_size=(1, 2))

        self.conv1_2 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 15), padding=(0, 7))
        self.norm1_2 = nn.GroupNorm(4, out_chans)
        self.pool1_2 = nn.AvgPool2d(kernel_size=(1, 2))

        self.conv1_3 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 9), padding=(0, 4))
        self.norm1_3 = nn.GroupNorm(4, out_chans)
        self.pool1_3 = nn.AvgPool2d(kernel_size=(1, 2))

        self.conv1_4 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 5), padding=(0, 2))
        self.norm1_4 = nn.GroupNorm(4, out_chans)
        self.pool1_4 = nn.AvgPool2d(kernel_size=(1, 2))
        self.gelu1 = nn.GELU()

        self.conv2_1 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 9), padding=(0, 4))
        self.norm2_1 = nn.GroupNorm(4, out_chans)
        self.pool2_1 = nn.AvgPool2d(kernel_size=(1, 4))

        self.conv2_2 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 7), padding=(0, 3))
        self.norm2_2 = nn.GroupNorm(4, out_chans)
        self.pool2_2 = nn.AvgPool2d(kernel_size=(1, 4))

        self.conv2_3 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 5), padding=(0, 2))
        self.norm2_3 = nn.GroupNorm(4, out_chans)
        self.pool2_3 = nn.AvgPool2d(kernel_size=(1, 4))

        self.conv2_4 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.norm2_4 = nn.GroupNorm(4, out_chans)
        self.pool2_4 = nn.AvgPool2d(kernel_size=(1, 4))
        self.gelu2 = nn.GELU()

    def forward(self, x):
        x = rearrange(x, "B N A T -> B (N A) T")
        x = x.unsqueeze(1)

        x1 = self.pool1_1(self.gelu1(self.norm1_1(self.conv1_1(x))))
        x2 = self.pool1_2(self.gelu1(self.norm1_2(self.conv1_2(x))))
        x3 = self.pool1_3(self.gelu1(self.norm1_3(self.conv1_3(x))))
        x4 = self.pool1_4(self.gelu1(self.norm1_4(self.conv1_4(x))))

        x1 = self.pool2_1(self.gelu2(self.norm2_1(self.conv2_1(x1))))
        x2 = self.pool2_2(self.gelu2(self.norm2_2(self.conv2_2(x2))))
        x3 = self.pool2_3(self.gelu2(self.norm2_3(self.conv2_3(x3))))
        x4 = self.pool2_4(self.gelu2(self.norm2_4(self.conv2_4(x4))))

        x1 = rearrange(x1, "B C NA T -> B NA (T C)")
        x2 = rearrange(x2, "B C NA T -> B NA (T C)")
        x3 = rearrange(x3, "B C NA T -> B NA (T C)")
        x4 = rearrange(x4, "B C NA T -> B NA (T C)")
        return x1, x2, x3, x4


@dataclass
class NeuroRVQBackboneConfig:
    n_patches: int = 256
    patch_size: int = 200
    out_chans: int = 8
    embed_dim: int = 200
    depth: int = 12
    num_heads: int = 10
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    init_values: float = 1e-5
    init_scale: float = 0.001
    n_global_electrodes: int = 104


class NeuroRVQBackbone(nn.Module):
    """Feature extractor path of NeuroRVQ foundation model (EEG variant)."""

    def __init__(self, config: NeuroRVQBackboneConfig):
        super().__init__()
        self.config = config

        self.patch_embed = MultiDimensionalTemporalConv(out_chans=config.out_chans)
        self.pos_embed = nn.Parameter(torch.zeros(config.n_global_electrodes + 1, config.embed_dim), requires_grad=True)
        self.time_embed = nn.Parameter(torch.zeros(config.n_patches, config.embed_dim), requires_grad=True)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.pos_drop = nn.Dropout(p=config.drop_rate)

        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=config.embed_dim,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    qkv_bias=config.qkv_bias,
                    qk_norm=partial(nn.LayerNorm, eps=1e-6),
                    drop=config.drop_rate,
                    attn_drop=config.attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=nn.LayerNorm,
                    init_values=config.init_values,
                )
                for i in range(config.depth)
            ]
        )
        self.norm = nn.Identity()
        self.norm_pre = nn.LayerNorm(config.embed_dim)

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.time_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.mask_token, std=0.02)

        self.apply(self._init_weights)
        self._fix_init_weight()

    def _fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv2d):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def _encode_single_branch(self, branch_tokens, spatial_embedding, temporal_embedding):
        branch_tokens = branch_tokens + spatial_embedding
        branch_tokens[:, 1:, :] += temporal_embedding
        branch_tokens = self.pos_drop(branch_tokens)

        for block in self.blocks:
            branch_tokens = block(branch_tokens)

        branch_tokens = self.norm(branch_tokens)
        return branch_tokens[:, 1:, :]

    def forward_features(self, x, temporal_embedding_ix, spatial_embedding_ix):
        """Return pooled representation with shape [B, 4 * embed_dim]."""
        x1, x2, x3, x4 = self.patch_embed(x)
        batch_size = x1.shape[0]

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x1 = torch.cat((cls_tokens, x1), dim=1)
        x2 = torch.cat((cls_tokens, x2), dim=1)
        x3 = torch.cat((cls_tokens, x3), dim=1)
        x4 = torch.cat((cls_tokens, x4), dim=1)

        spatial_embedding_ix = F.pad(input=spatial_embedding_ix, pad=(1, 0), mode="constant", value=0)
        spatial_embedding = self.pos_embed[spatial_embedding_ix.reshape(-1), :]
        spatial_embedding = spatial_embedding.reshape(
            spatial_embedding_ix.shape[0],
            spatial_embedding_ix.shape[1],
            spatial_embedding.shape[-1],
        )

        temporal_embedding = self.time_embed[temporal_embedding_ix.reshape(-1), :]
        temporal_embedding = temporal_embedding.reshape(
            temporal_embedding_ix.shape[0],
            temporal_embedding_ix.shape[1],
            temporal_embedding.shape[-1],
        )

        x1 = self._encode_single_branch(x1, spatial_embedding, temporal_embedding)
        x2 = self._encode_single_branch(x2, spatial_embedding, temporal_embedding)
        x3 = self._encode_single_branch(x3, spatial_embedding, temporal_embedding)
        x4 = self._encode_single_branch(x4, spatial_embedding, temporal_embedding)

        fused = torch.cat([x1, x2, x3, x4], dim=-1)
        return fused.mean(dim=1)

    def forward(self, x, temporal_embedding_ix, spatial_embedding_ix):
        return self.forward_features(x, temporal_embedding_ix, spatial_embedding_ix)
