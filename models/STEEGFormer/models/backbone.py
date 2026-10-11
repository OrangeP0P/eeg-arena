"""Minimal ST-EEGFormer backbone adapted for the unified framework."""
from functools import partial
import math

import torch
import torch.nn as nn
import torch.nn.init as init
import timm.models.vision_transformer


class PatchEmbedEEG(nn.Module):
    """Patchify raw EEG along the temporal axis and project each patch."""

    def __init__(self, patch_size=16, embed_dim=256):
        super().__init__()
        self.patch_size = int(patch_size)
        self.unfold = torch.nn.Unfold(
            kernel_size=(1, self.patch_size),
            stride=self.patch_size,
        )
        self.proj = nn.Linear(self.patch_size, embed_dim)

    def forward(self, x):
        return self.proj(self.patchify_eeg(x))

    def patchify_eeg(self, x):
        batch_size, channels, time_len = x.shape
        del time_len
        unfolded = self.unfold(x.unsqueeze(2))
        _, _, seq_len = unfolded.shape
        unfolded = unfolded.reshape(batch_size, channels, self.patch_size, seq_len)
        return unfolded.permute(0, 3, 1, 2)


class ChannelPositionalEmbed(nn.Module):
    """Learned channel embedding used by ST-EEGFormer."""

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.channel_transformation = nn.Embedding(num_embeddings, embedding_dim)
        init.zeros_(self.channel_transformation.weight)

    def forward(self, channel_indices):
        return self.channel_transformation(channel_indices)


class TemporalPositionalEncoding(nn.Module):
    """Sinusoidal temporal encoding over patch indices."""

    def __init__(self, d_model, max_len=512):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            (torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)).float()
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position.float() * div_term)
        pe[0, :, 1::2] = torch.cos(position.float() * div_term)
        self.register_buffer("pe", pe)

    def get_cls_token(self):
        return self.pe[0, 0, :]

    def forward(self, seq_indices):
        batch_size, seq_len = seq_indices.shape
        return self.pe[0, seq_indices.reshape(-1)].reshape(batch_size, seq_len, -1)


class STEEGFormerVisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """Upstream ST-EEGFormer encoder with configurable channel embedding size."""

    def __init__(self, global_pool=True, num_channel_embeddings=145, **kwargs):
        super().__init__(**kwargs)
        self.global_pool = bool(global_pool)

        if self.global_pool:
            norm_layer = kwargs["norm_layer"]
            embed_dim = kwargs["embed_dim"]
            self.fc_norm = norm_layer(embed_dim)
            del self.norm

        self.patch_embed = PatchEmbedEEG(
            patch_size=kwargs["patch_size"],
            embed_dim=kwargs["embed_dim"],
        )
        self.enc_channel_emd = ChannelPositionalEmbed(
            num_embeddings=num_channel_embeddings,
            embedding_dim=kwargs["embed_dim"],
        )
        self.enc_temporal_emd = TemporalPositionalEncoding(
            kwargs["embed_dim"],
            max_len=512,
        )

    def _prepare_channel_indices(self, chan_idx, batch_size, num_channels, device):
        if chan_idx is None:
            raise ValueError("STEEGFormer requires channel indices for every input channel.")

        if chan_idx.dim() == 1:
            chan_idx = chan_idx.unsqueeze(0)
        if chan_idx.dim() != 2:
            raise ValueError(
                f"STEEGFormer channel indices must be 1D or 2D, got shape {tuple(chan_idx.shape)}."
            )
        if chan_idx.shape[0] == 1 and batch_size != 1:
            chan_idx = chan_idx.expand(batch_size, -1)
        if chan_idx.shape[0] != batch_size or chan_idx.shape[1] != num_channels:
            raise ValueError(
                "STEEGFormer channel indices do not match the EEG batch shape: "
                f"indices={tuple(chan_idx.shape)}, eeg=({batch_size}, {num_channels}, ...)."
            )

        return chan_idx.to(device=device, dtype=torch.long)

    def forward_features(self, eeg, chan_idx):
        x = self.patch_embed(eeg)
        batch_size, seq_len, num_channels, embed_dim = x.shape
        x = x.reshape(batch_size, seq_len * num_channels, embed_dim)

        chan_idx = self._prepare_channel_indices(chan_idx, batch_size, num_channels, eeg.device)
        eeg_chan_indices = chan_idx.unsqueeze(1).expand(batch_size, seq_len, num_channels)
        eeg_chan_indices = eeg_chan_indices.reshape(batch_size, seq_len * num_channels)

        seq_tensor = torch.arange(1, seq_len + 1, device=eeg.device, dtype=torch.long)
        eeg_seq_indices = seq_tensor.view(1, seq_len, 1).expand(batch_size, seq_len, num_channels)
        eeg_seq_indices = eeg_seq_indices.reshape(batch_size, seq_len * num_channels)

        x = x + self.enc_temporal_emd(eeg_seq_indices) + self.enc_channel_emd(eeg_chan_indices)

        cls_token = self.cls_token + self.enc_temporal_emd.get_cls_token().view(1, 1, -1)
        cls_tokens = cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        if self.global_pool:
            return x[:, 1:, :].mean(dim=1)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, eeg, chan_idx):
        x = self.forward_features(eeg, chan_idx)
        x = self.head_drop(x)
        return self.head(x)


def build_steegformer_backbone(
    *,
    embed_dim,
    depth,
    num_heads,
    num_classes,
    classifier_dropout=0.1,
    num_channel_embeddings=145,
    global_pool=True,
    attn_drop_rate=0.1,
    proj_drop_rate=0.1,
    drop_path_rate=0.1,
):
    """Build an ST-EEGFormer backbone matching the upstream checkpoints."""

    return STEEGFormerVisionTransformer(
        patch_size=16,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        num_classes=num_classes,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        global_pool=global_pool,
        num_channel_embeddings=num_channel_embeddings,
        drop_rate=classifier_dropout,
        attn_drop_rate=attn_drop_rate,
        proj_drop_rate=proj_drop_rate,
        drop_path_rate=drop_path_rate,
    )
