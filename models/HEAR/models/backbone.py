import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from einops import rearrange
import numpy as np

from .channel_info import LOCATION_STANDARD_1020

# from wh_v1.modeling_pretrain_old import NeuralTransformerForMEM


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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
        # x = self.drop(x)
        # commit this for the orignal BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_norm=None, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

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

        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
            # cls to token & token 2 cls & cls to cls

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(window_size[0])
            coords_w = torch.arange(window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = \
                torch.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
            relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1

            self.register_buffer("relative_position_index", relative_position_index)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos_bias=None, return_attention=False, return_qkv=False):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple) (B, H, N, C)
        if self.q_norm is not None:
            q = self.q_norm(q).type_as(v)
        if self.k_norm is not None:
            k = self.k_norm(k).type_as(v)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.relative_position_bias_table is not None:
            relative_position_bias = \
                self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                    self.window_size[0] * self.window_size[1] + 1,
                    self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if return_attention:
            return attn

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)

        if return_qkv:
            return x, qkv

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_norm=None, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, rel_pos_bias=None, return_attention=False, return_qkv=False):
        if return_attention:
            return self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, return_attention=True)
        if return_qkv:
            y, qkv = self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, return_qkv=return_qkv)
            x = x + self.drop_path(self.gamma_1 * y)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
            return x, qkv

        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ EEG to Patch Embedding
    """

    def __init__(self, EEG_size=2000, patch_size=200, in_chans=1, embed_dim=200):
        super().__init__()
        # EEG_size = to_2tuple(EEG_size)
        # patch_size = to_2tuple(patch_size)
        num_patches = 62 * (EEG_size // patch_size)
        self.patch_shape = (1, EEG_size // patch_size)
        self.EEG_size = EEG_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=(1, patch_size), stride=(1, patch_size))

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class TemporalConv(nn.Module):
    """ EEG to Patch Embedding
    """

    def __init__(self, in_chans=1, out_chans=8):
        '''
        in_chans: in_chans of nn.Conv2d()
        out_chans: out_chans of nn.Conv2d(), determing the output dimension
        '''
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 15), stride=(1, 8), padding=(0, 7))
        self.gelu1 = nn.GELU()
        self.norm1 = nn.GroupNorm(4, out_chans)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.gelu2 = nn.GELU()
        self.norm2 = nn.GroupNorm(4, out_chans)
        self.conv3 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.norm3 = nn.GroupNorm(4, out_chans)
        self.gelu3 = nn.GELU()

    def forward(self, x, **kwargs):
        x = rearrange(x, 'B N A T -> B (N A) T')
        B, NA, T = x.shape
        x = x.unsqueeze(1)
        x = self.gelu1(self.norm1(self.conv1(x)))
        x = self.gelu2(self.norm2(self.conv2(x)))
        x = self.gelu3(self.norm3(self.conv3(x)))
        x = rearrange(x, 'B C NA T -> B NA (T C)')
        return x


class SpatialEmbedding(nn.Module):
    def __init__(self, input_dim=3, embed_dim=200):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim)
        )

    def forward(self, coords):
        return self.embed(coords)  # [num_channels, embed_dim]


class NeuralTransformer(nn.Module):
    def __init__(self, EEG_size=1600, patch_size=200, in_chans=1, out_chans=8, num_classes=1000, embed_dim=200,
                 depth=12,
                 num_heads=10, mlp_ratio=4., qkv_bias=False, qk_norm=None, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False,
                 use_mean_pooling=True, init_scale=0.001, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_heads = num_heads
        # To identify whether it is neural tokenizer or neural decoder.
        # For the neural decoder, use linear projection (PatchEmbed) to project codebook dimension to hidden dimension.
        # Otherwise, use TemporalConv to extract temporal features from EEG signals.
        self.patch_embed = TemporalConv(out_chans=out_chans) if in_chans == 1 else PatchEmbed(EEG_size=EEG_size,
                                                                                              patch_size=patch_size,
                                                                                              in_chans=in_chans,
                                                                                              embed_dim=embed_dim)
        self.time_window = EEG_size // patch_size
        self.patch_size = patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Fixed upstream standard_1020 coordinates are kept as a fallback. The
        # default fine-tuning path can pass runtime coordinates for arbitrary
        # heterogeneous channel layouts.
        coords = torch.tensor(LOCATION_STANDARD_1020, dtype=torch.float32)  # shape [C, 3]
        self.register_buffer("electrode_coords", coords, persistent=False)
        self.coords = self.electrode_coords
        self.attn_weights_all = []
        self.record_attn = True  # 默认不记录，eval 时可以开启

        # 空间编码修改
        if use_abs_pos_emb:
            self.spatial_encoder = SpatialEmbedding(input_dim=3, embed_dim=embed_dim)
            with torch.no_grad():
                self.register_buffer("pos_embed", self.spatial_encoder(self.electrode_coords))
        else:
            self.spatial_encoder = None
            self.pos_embed = None

        if use_rel_pos_bias:
            self.spatial_bias = 1
        else:
            self.spatial_bias = None
            self.rel_pos_bias = None

        self.time_embed = nn.Parameter(torch.zeros(1, 128, embed_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.use_rel_pos_bias = use_rel_pos_bias
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_norm=qk_norm,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=None)
            for i in range(depth)])
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        # self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 4),
            nn.ELU(),  # 或 GELU
            nn.Dropout(0.5),  # 适当增加 Dropout 防止过拟合
            nn.Linear(self.embed_dim * 4, self.embed_dim),
            nn.ELU(),
            nn.Linear(self.embed_dim, num_classes)
        )

        # —— Relative Spatial Bias MLP ——
        # 用来把 3D 坐标差映射到每个 head 的 bias
        self.elec_mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        hidden = embed_dim // 2
        self.spatial_bias_mlp = nn.Sequential(
            nn.Linear(3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_heads)
        )

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        if self.time_embed is not None:
            trunc_normal_(self.time_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        # trunc_normal_(self.mask_token, std=.02)
        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=.02)
        self.apply(self._init_weights)
        self.fix_init_weight()

        if isinstance(self.head, nn.Linear):
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _resolve_runtime_coords(self, x, input_chans=None, input_coords=None, num_channels=None):
        num_channels = int(num_channels if num_channels is not None else x.shape[1])
        if input_coords is not None:
            coords = torch.as_tensor(input_coords, dtype=x.dtype, device=x.device)
            if coords.ndim == 3:
                coords = coords[0]
            if coords.ndim != 2 or coords.shape[-1] != 3:
                raise ValueError(
                    "HEAR input_coords must have shape [channels, 3] "
                    f"or [batch, channels, 3], got {tuple(coords.shape)}."
                )
            if coords.shape[0] < num_channels:
                raise ValueError(
                    f"HEAR received {coords.shape[0]} coordinate rows for "
                    f"{num_channels} input channels."
                )
            return coords[:num_channels]

        coords_tensor = self.electrode_coords.to(device=x.device, dtype=x.dtype)
        if input_chans is None:
            if num_channels > coords_tensor.shape[0]:
                raise ValueError(
                    f"HEAR received {num_channels} channels, but fixed standard_1020 "
                    f"coordinates only contain {coords_tensor.shape[0]} entries."
                )
            input_chans = [0] + list(range(num_channels))
        input_idx = torch.tensor(input_chans[1:], dtype=torch.long, device=x.device)
        return coords_tensor[input_idx]

    def forward_features(self, x, input_chans=None, input_coords=None, return_patch_tokens=False, return_all_tokens=False, **kwargs):
        batch_size, n, a, t = x.shape
        batch_size, c, time_window, _ = x.size()
        input_time_window = a if t == self.patch_size else t
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        coords_used = None
        if self.spatial_encoder is not None or self.spatial_bias is not None:
            coords_used = self._resolve_runtime_coords(
                x,
                input_chans=input_chans,
                input_coords=input_coords,
                num_channels=c,
            )

        # ---- Step 2: Spatial soft embedding ----
        if self.spatial_encoder is not None:
            # print('空间编码成功')
            spatial_embed = self.spatial_encoder(coords_used)  # [C, D]
            spatial_embed = spatial_embed.unsqueeze(1).expand(-1, input_time_window, -1)  # [C, T, D]
            spatial_embed = spatial_embed.flatten(0, 1).unsqueeze(0).expand(batch_size, -1, -1)  # [B, C×T, D]
            cls_spatial_embed = torch.zeros_like(self.cls_token).expand(batch_size, -1, -1)  # [B, 1, D]
            spatial_embed = torch.cat((cls_spatial_embed, spatial_embed), dim=1)  # [B, 1 + C×T, D]
            x = x + spatial_embed

        if self.time_embed is not None:
            nc = n if t == self.patch_size else a
            time_embed = self.time_embed[:, 0:input_time_window, :].unsqueeze(1).expand(batch_size, nc, -1, -1).flatten(
                1, 2)
            x[:, 1:, :] += time_embed

        x = self.pos_drop(x)

        if self.spatial_bias is not None:
            # print('空间 bias 建模成功')
            cls_tok, patches = x[:, :1, :], x[:, 1:, :]   # cls_tok [B,1,D], patches [B, C*T, D]
            patches = patches.view(batch_size, c, input_time_window, -1)  # [B, C, T, D]
            patches = patches.permute(0, 2, 1, 3).reshape(batch_size * input_time_window, c, -1)  # [B*T, C, D]
            elec_out, attn_weights = self.elec_mha(patches, patches, patches, need_weights=True)  # [B*T, C, D]

            if self.record_attn:  # 记录 attention_weights
                # attn_weights: [B*T, H, C, C]
                self.attn_weights_all.append(attn_weights.detach().cpu())

            elec_out = elec_out.reshape(batch_size, input_time_window, c, -1)  # [B, T, C, D]
            elec_out = elec_out.permute(0, 2, 1, 3).reshape(batch_size, c * input_time_window, -1)  # [B, C*T, D]
            x = torch.cat([cls_tok, elec_out], dim=1)  # [B, 1+C*T, D]

            # —— 计算 Relative Spatial Bias ——
            # 1) 两两坐标差 [C,C,3]
            rel_coords = coords_used[:, None, :] - coords_used[None, :, :]

            # 2) MLP 映射到每个 head 的 bias [C,C,H]
            bias_cc_h = self.spatial_bias_mlp(rel_coords)         # [C,C,num_heads]
            bias_h_cc = bias_cc_h.permute(2, 0, 1).contiguous()     # [num_heads,C,C]

            # 3) 按时间窗口重复，扩展到 [num_heads, C*T, C*T]
            bias_h_ctct = bias_h_cc.repeat_interleave(input_time_window, 1) \
                                 .repeat_interleave(input_time_window, 2)   # [H, C*T, C*T]

            # 4) 构造完整的 rel_pos_bias 张量并填入 [1,H,N,N]
            CT = c * input_time_window
            N = 1 + CT
            rel_pos_bias = x.new_zeros(1, self.num_heads, N, N)   # [1,H,N,N]
            rel_pos_bias[0, :, 1:, 1:] = bias_h_ctct              # 保持 cls token 的行列为 0

            # —— 传入各 Block ——
            for blk in self.blocks:
                x = blk(x, rel_pos_bias=rel_pos_bias)
        else:
            # —— 传入各 Block ——
            for blk in self.blocks:
                x = blk(x, rel_pos_bias=None)

        x = self.norm(x)
        if self.fc_norm is not None:
            if return_all_tokens:
                return self.fc_norm(x)
            t = x[:, 1:, :]
            if return_patch_tokens:
                return self.fc_norm(t)
            else:
                return self.fc_norm(t.mean(1))
        else:
            if return_all_tokens:
                return x
            elif return_patch_tokens:
                return x[:, 1:]
            else:
                return x[:, 0]

    def forward(self, x, input_chans=None, input_coords=None, return_patch_tokens=False, return_all_tokens=False, **kwargs):
        '''
        x: [batch size, number of electrodes, number of patches, patch size]
        For example, for an EEG sample of 4 seconds with 64 electrodes, x will be [batch size, 64, 4, 200]
        '''
        x = self.forward_features(x, input_chans=input_chans, input_coords=input_coords, return_patch_tokens=return_patch_tokens,
                                  return_all_tokens=return_all_tokens, **kwargs)
        x = self.head(x)
        return x

    def forward_intermediate(self, x, layer_id=12, norm_output=False):
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        # print('第二次编码成功')

        if self.pos_embed is not None:
            print('第二次编码成功')
            pos_embed = self.pos_embed[:, 1:, :].unsqueeze(2).expand(batch_size, -1, self.time_window, -1).flatten(1, 2)
            pos_embed = torch.cat((self.pos_embed[:, 0:1, :].expand(batch_size, -1, -1), pos_embed), dim=1)
            x = x + pos_embed
        if self.time_embed is not None:
            time_embed = self.time_embed.unsqueeze(1).expand(batch_size, 62, -1, -1).flatten(1, 2)
            x[:, 1:, :] += time_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        if isinstance(layer_id, list):
            output_list = []
            for l, blk in enumerate(self.blocks):
                x = blk(x, rel_pos_bias=rel_pos_bias)
                # use last norm for all intermediate layers
                if l in layer_id:
                    if norm_output:
                        x_norm = self.fc_norm(self.norm(x[:, 1:]))
                        output_list.append(x_norm)
                    else:
                        output_list.append(x[:, 1:])
            return output_list

        elif isinstance(layer_id, int):
            for l, blk in enumerate(self.blocks):
                if l < layer_id:
                    x = blk(x, rel_pos_bias=rel_pos_bias)
                elif l == layer_id:
                    x = blk.norm1(x)
                else:
                    break
            return x[:, 1:]
        else:
            raise NotImplementedError(f"Not support for layer id is {layer_id} now!")

    def get_intermediate_layers(self, x, use_last_norm=False):
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        # print('第三次编码成功')
        if self.pos_embed is not None:
            pos_embed = self.pos_embed[:, 1:, :].unsqueeze(2).expand(batch_size, -1, self.time_window, -1).flatten(1, 2)
            pos_embed = torch.cat((self.pos_embed[:, 0:1, :].expand(batch_size, -1, -1), pos_embed), dim=1)
            x = x + pos_embed
        if self.time_embed is not None:
            time_embed = self.time_embed.unsqueeze(1).expand(batch_size, 62, -1, -1).flatten(1, 2)
            x[:, 1:, :] += time_embed
        x = self.pos_drop(x)

        features = []
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias)
            if use_last_norm:
                features.append(self.norm(x))
            else:
                features.append(x)

        return features

# @register_model
# def tiny(pretrained=False, **kwargs):
#     model = NeuralTransformer(
#         patch_size=200, embed_dim=200, depth=6, num_heads=4, mlp_ratio=4, qk_norm=partial(nn.LayerNorm, eps=1e-6), # qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     model.default_cfg = _cfg()
#     return model
#
# @register_model
# def small(pretrained=False, **kwargs):
#     model = NeuralTransformer(
#         patch_size=200, embed_dim=200, depth=12, num_heads=8, mlp_ratio=4, qk_norm=partial(nn.LayerNorm, eps=1e-6), # qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     model.default_cfg = _cfg()
#     return model
#
# @register_model
# def base(pretrained=False, **kwargs):
#     model = NeuralTransformer(
#         patch_size=200, embed_dim=200, depth=24, num_heads=20, mlp_ratio=4, qk_norm=partial(nn.LayerNorm, eps=1e-6), # qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     model.default_cfg = _cfg()
#     return model
#
# @register_model
# def large(pretrained=False, **kwargs):
#     model = NeuralTransformer(
#         patch_size=200, embed_dim=400, depth=48, num_heads=25, mlp_ratio=4, qk_norm=partial(nn.LayerNorm, eps=1e-6), out_chans=16, # qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     model.default_cfg = _cfg()
#     return model


import torch
import torch.nn as nn
from functools import partial

@register_model
def M1(pretrained=False, **kwargs):
    """
    Data: ~75% | Depth: 18 | Params: ~8.6M
    """
    vocab_size = kwargs.pop("vocab_size", 8192)
    if "num_classes" in kwargs: kwargs.pop("num_classes")

    model = NeuralTransformer(
        patch_size=200, embed_dim=200, depth=18, num_heads=20, mlp_ratio=4, vocab_size=vocab_size,
        qk_norm=partial(nn.LayerNorm, eps=1e-6),
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    model.default_cfg = _cfg()

    if pretrained and "init_ckpt" in kwargs:
        checkpoint = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])

    return model


@register_model
def M2(pretrained=False, **kwargs):
    """
    Data: ~58% | Depth: 14 | Params: ~6.7M
    """
    vocab_size = kwargs.pop("vocab_size", 8192)
    if "num_classes" in kwargs: kwargs.pop("num_classes")

    model = NeuralTransformer(
        patch_size=200, embed_dim=200, depth=14, num_heads=20, mlp_ratio=4,
        vocab_size=vocab_size,
        qk_norm=partial(nn.LayerNorm, eps=1e-6),
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    model.default_cfg = _cfg()

    if pretrained and "init_ckpt" in kwargs:
        checkpoint = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])

    return model


@register_model
def M3(pretrained=False, **kwargs):
    """
    Data: ~41% | Depth: 10 | Params: ~4.8M
    """
    vocab_size = kwargs.pop("vocab_size", 8192)
    if "num_classes" in kwargs: kwargs.pop("num_classes")

    model = NeuralTransformer(
        patch_size=200, embed_dim=200, depth=10, num_heads=20, mlp_ratio=4,
        vocab_size=vocab_size,
        qk_norm=partial(nn.LayerNorm, eps=1e-6),
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    model.default_cfg = _cfg()

    if pretrained and "init_ckpt" in kwargs:
        checkpoint = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])

    return model


@register_model
def M4(pretrained=False, **kwargs):
    """
    Data: ~29% | Depth: 7 | Params: ~3.4M
    """
    vocab_size = kwargs.pop("vocab_size", 8192)
    if "num_classes" in kwargs: kwargs.pop("num_classes")

    model = NeuralTransformer(
        patch_size=200, embed_dim=200, depth=7, num_heads=20, mlp_ratio=4,
        vocab_size=vocab_size,
        qk_norm=partial(nn.LayerNorm, eps=1e-6),
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    model.default_cfg = _cfg()

    if pretrained and "init_ckpt" in kwargs:
        checkpoint = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])

    return model


@register_model
def M5(pretrained=False, **kwargs):
    """
    Data: ~20% | Depth: 5 | Params: ~2.4M
    """
    vocab_size = kwargs.pop("vocab_size", 8192)
    if "num_classes" in kwargs: kwargs.pop("num_classes")

    model = NeuralTransformer(
        patch_size=200, embed_dim=200, depth=5, num_heads=20, mlp_ratio=4,
        vocab_size=vocab_size,
        qk_norm=partial(nn.LayerNorm, eps=1e-6),
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    model.default_cfg = _cfg()

    if pretrained and "init_ckpt" in kwargs:
        checkpoint = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])

    return model


@register_model
def M6(pretrained=False, **kwargs):
    """
    Data: ~12% | Depth: 3 | Params: ~1.4M
    """
    vocab_size = kwargs.pop("vocab_size", 8192)
    if "num_classes" in kwargs: kwargs.pop("num_classes")

    model = NeuralTransformer(
        patch_size=200, embed_dim=200, depth=3, num_heads=20, mlp_ratio=4,
        vocab_size=vocab_size,
        qk_norm=partial(nn.LayerNorm, eps=1e-6),
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    model.default_cfg = _cfg()

    if pretrained and "init_ckpt" in kwargs:
        checkpoint = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])

    return model


@register_model
def M7(pretrained=False, **kwargs):
    """
    Data: ~8.3% | Depth: 2 | Params: ~0.96M
    """
    vocab_size = kwargs.pop("vocab_size", 8192)
    if "num_classes" in kwargs: kwargs.pop("num_classes")

    model = NeuralTransformer(
        patch_size=200, embed_dim=200, depth=2, num_heads=20, mlp_ratio=4,
        vocab_size=vocab_size,
        qk_norm=partial(nn.LayerNorm, eps=1e-6),
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    model.default_cfg = _cfg()

    if pretrained and "init_ckpt" in kwargs:
        checkpoint = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])

    return model
