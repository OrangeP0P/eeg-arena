"""LUNA backbone adapted for unified fine-tuning."""
import math

import torch
import torch.nn as nn
from einops import rearrange
from timm.layers import Mlp, trunc_normal_

from .modules import FrequencyFeatureEmbedder, RotaryTransformerBlock


def nerf_positional_encoding(coords: torch.Tensor, embed_size: int) -> torch.Tensor:
    """Apply NeRF-style positional encoding to 3D coordinates."""
    batch, channels, dim = coords.shape
    device = coords.device
    freqs = embed_size // (2 * dim)
    leftover = embed_size - freqs * 2 * dim

    freq_bands = 2.0 ** torch.arange(freqs, device=device).float()
    scaled = coords.unsqueeze(-1) * freq_bands.view(1, 1, 1, -1)
    sin_enc = torch.sin(scaled)
    cos_enc = torch.cos(scaled)

    encoded = (
        torch.stack((sin_enc, cos_enc), dim=-1)
        .permute(0, 1, 3, 2, 4)
        .reshape(batch, channels, freqs * dim * 2)
    )

    if leftover > 0:
        pad = torch.zeros(batch, channels, leftover, device=device, dtype=coords.dtype)
        encoded = torch.cat((encoded, pad), dim=-1)
    return encoded


class ClassificationHeadWithQueries(nn.Module):
    """Query-based classifier used by LUNA fine-tuning."""

    def __init__(
        self,
        embed_dim=768,
        num_queries=8,
        num_heads=8,
        num_classes=2,
        dropout=0.15,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim * num_queries)
        self.decoder_attn = nn.MultiheadAttention(
            self.embed_dim,
            num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.decoder_ffn = Mlp(
            in_features=self.embed_dim,
            hidden_features=int(self.embed_dim * 4),
            out_features=num_classes,
            act_layer=nn.GELU,
            drop=dropout,
        )
        self.learned_agg = nn.Parameter(torch.randn(1, 1, self.embed_dim), requires_grad=True)

    def pool_features(self, latent_tokens):
        queries = self.learned_agg.repeat(latent_tokens.shape[0], 1, 1)
        pooled = self.decoder_attn(query=queries, key=latent_tokens, value=latent_tokens)[0]
        return pooled[:, 0, :]

    def forward(self, latent_tokens):
        features = self.pool_features(latent_tokens)
        return self.decoder_ffn(features)


class CrossAttentionBlock(nn.Module):
    """Cross-attention that maps channel tokens into query tokens."""

    def __init__(
        self,
        num_queries,
        input_embed_dim,
        output_embed_dim,
        num_heads,
        dropout_p=0.1,
        ff_dim=2048,
    ):
        super().__init__()
        self.query_embed = nn.Parameter(
            torch.randn(1, num_queries, input_embed_dim),
            requires_grad=True,
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=input_embed_dim,
            num_heads=num_heads,
            dropout=dropout_p,
            batch_first=True,
        )
        self.ffn = Mlp(
            input_embed_dim,
            ff_dim,
            output_embed_dim,
            act_layer=nn.GELU,
            drop=dropout_p,
            norm_layer=nn.LayerNorm,
        )
        self.keys_norm = nn.LayerNorm(input_embed_dim)
        self.values_norm = nn.LayerNorm(input_embed_dim)
        self.queries_norm = nn.LayerNorm(input_embed_dim)
        self.query_self_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                input_embed_dim,
                nhead=num_heads,
                activation="gelu",
                dim_feedforward=ff_dim,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=3,
        )

    def initialize_weights(self):
        nn.init.orthogonal_(self.query_embed, gain=1.0)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, x):
        batch = x.size(0)
        queries = self.queries_norm(self.query_embed.repeat(batch, 1, 1))
        keys = self.keys_norm(x)
        values = self.values_norm(x)

        attention_out, attention_scores = self.cross_attention(
            query=queries,
            key=keys,
            value=values,
        )
        attention_out = self.ffn(attention_out) + attention_out
        attention_out = self.query_self_attn(attention_out)
        return attention_out, attention_scores


class PatchEmbedNetwork(nn.Module):
    """Temporal patch embedder from raw EEG."""

    def __init__(self, embed_dim=64, patch_size=40):
        super().__init__()
        out_channels = int(embed_dim // 4)
        kernel_size = int(patch_size // 2)
        groups = 4

        self.patch_size = int(patch_size)
        self.proj_in = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=out_channels,
                kernel_size=(1, kernel_size - 1),
                stride=(1, kernel_size // 2),
                padding=(0, kernel_size // 2 - 1),
            ),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(1, 3),
                stride=(1, 1),
                padding=(0, 1),
            ),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(1, 3),
                stride=(1, 1),
                padding=(0, 1),
            ),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        x = rearrange(x, "B C (S P) -> B (C S) P", P=self.patch_size)
        x = x.unsqueeze(1)
        x = self.proj_in(x)
        x = rearrange(x, "B E CS D -> B CS (D E)")
        return x


class LUNABackbone(nn.Module):
    """LUNA backbone with classification head."""

    def __init__(
        self,
        patch_size=40,
        num_queries=4,
        embed_dim=64,
        depth=8,
        num_heads=2,
        mlp_ratio=4.0,
        drop_path=0.0,
        num_classes=2,
        classifier_dropout=0.15,
    ):
        super().__init__()
        if num_classes <= 0:
            raise ValueError("LUNABackbone requires num_classes > 0 for fine-tuning.")

        self.embed_dim = int(embed_dim)
        self.num_queries = int(num_queries)
        self.patch_size = int(patch_size)
        self.num_heads = int(num_heads)
        self.num_classes = int(num_classes)

        self.patch_embed = PatchEmbedNetwork(embed_dim=self.embed_dim, patch_size=self.patch_size)
        self.freq_embed = FrequencyFeatureEmbedder(embed_dim=self.embed_dim, patch_size=self.patch_size)
        self.channel_location_embedder = nn.Sequential(
            Mlp(
                in_features=self.embed_dim,
                out_features=self.embed_dim,
                hidden_features=int(self.embed_dim * 2),
                act_layer=nn.GELU,
                drop=0.0,
                norm_layer=nn.LayerNorm,
            ),
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.cross_attn = CrossAttentionBlock(
            num_queries=self.num_queries,
            input_embed_dim=self.embed_dim,
            output_embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            ff_dim=int(mlp_ratio * self.embed_dim),
        )
        self.blocks = nn.ModuleList(
            [
                RotaryTransformerBlock(
                    dim=int(self.embed_dim * self.num_queries),
                    num_heads=int(self.num_heads * self.num_queries),
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    drop=0.0,
                    attn_drop=0.0,
                    drop_path=drop_path,
                    norm_layer=nn.LayerNorm,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(int(self.embed_dim * self.num_queries))
        self.classifier = ClassificationHeadWithQueries(
            embed_dim=self.embed_dim,
            num_queries=self.num_queries,
            num_heads=self.num_heads,
            num_classes=self.num_classes,
            dropout=classifier_dropout,
        )

        self.mask_token.requires_grad = False
        self.initialize_weights()

    def initialize_weights(self):
        self.cross_attn.initialize_weights()
        trunc_normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)
        self.fix_init_weight()

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def prepare_tokens(self, x_signal, channel_locations, mask=None):
        num_channels = channel_locations.shape[1]
        num_patches_per_channel = x_signal.shape[-1] // self.patch_size

        x_patched = self.patch_embed(x_signal)
        x_patched = x_patched + self.freq_embed(x_signal)

        x_masked = x_patched
        if mask is not None:
            mask_tokens = self.mask_token.repeat(x_masked.shape[0], x_masked.shape[1], 1)
            patch_mask = rearrange(mask, "B C (S P) -> B (C S) P", P=self.patch_size)
            patch_mask = (patch_mask.sum(dim=-1) > 0).unsqueeze(-1)
            x_masked = torch.where(patch_mask, mask_tokens, x_masked)

        channel_min = torch.min(channel_locations, dim=1, keepdim=True)[0]
        channel_max = torch.max(channel_locations, dim=1, keepdim=True)[0]
        normalized_locations = (channel_locations - channel_min) / (channel_max - channel_min + 1e-8)

        location_embed = nerf_positional_encoding(normalized_locations, self.embed_dim)
        location_embed = self.channel_location_embedder(location_embed)

        x_tokenized = rearrange(x_masked, "B (C S) D -> (B S) C D", C=num_channels)
        location_embed = location_embed.repeat(num_patches_per_channel, 1, 1)
        x_tokenized = x_tokenized + location_embed
        return x_tokenized

    def encode(self, x_signal, channel_locations, mask=None):
        batch = x_signal.shape[0]
        x = self.prepare_tokens(x_signal, channel_locations, mask=mask)

        x, _ = self.cross_attn(x)
        x = rearrange(x, "(B S) Q D -> B S (Q D)", B=batch)

        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def extract_features(self, x_signal, channel_locations, mask=None):
        latent = self.encode(x_signal, channel_locations, mask=mask)
        return self.classifier.pool_features(latent)

    def forward(self, x_signal, mask, channel_locations):
        latent = self.encode(x_signal, channel_locations, mask=mask)
        return self.classifier(latent), x_signal
