"""Embedding layers used by LEADv2."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .augmentations import get_augmentation


SAMPLING_RATE_TO_ID = {200: 0, 100: 1, 50: 2}
NUM_SAMPLING_RATES = len(SAMPLING_RATE_TO_ID)


class Electrode3DEmbedding(nn.Module):
    """Sinusoidal 3D electrode embedding used by LEADv2."""

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.d_x = d_model // 3
        self.d_y = d_model // 3
        self.d_z = d_model - self.d_x - self.d_y

    def _encode_axis(self, pos, dim):
        device = pos.device
        pos = pos.unsqueeze(1)
        basis = torch.arange(0, dim, 2, device=device).float()
        div_term = torch.exp(-math.log(10000.0) * basis / dim)
        angles = pos * div_term

        emb = torch.zeros(pos.shape[0], dim, device=device)
        emb[:, 0::2] = torch.sin(angles)
        emb[:, 1::2] = torch.cos(angles)
        return emb

    def forward(self, coords):
        coords = coords.float()
        pe_x = self._encode_axis(coords[:, 0], self.d_x)
        pe_y = self._encode_axis(coords[:, 1], self.d_y)
        pe_z = self._encode_axis(coords[:, 2], self.d_z)
        return torch.cat([pe_x, pe_y, pe_z], dim=-1)


def build_ordered_channel_positions(ch_names, ch_positions, num_channels):
    """Return channel positions aligned to the runtime channel order."""
    positions = []
    position_dict = ch_positions if isinstance(ch_positions, dict) else {}
    for name in list(ch_names or [])[:num_channels]:
        pos = position_dict.get(name)
        if pos is None:
            pos = position_dict.get(f"EEG:{name}")
        if pos is None:
            pos = [0.0, 0.0, 0.0]
        pos = list(pos[:3]) if len(pos) >= 3 else list(pos) + [0.0] * (3 - len(pos))
        positions.append([float(value) for value in pos[:3]])

    while len(positions) < num_channels:
        positions.append([0.0, 0.0, 0.0])

    return torch.tensor(positions[:num_channels], dtype=torch.float32)


class LEADv2Embedding(nn.Module):
    """LEADv2 raw-signal patch embedding."""

    def __init__(
        self,
        enc_in,
        d_model,
        patch_len,
        stride,
        dropout,
        channel_positions,
        augmentations=None,
    ):
        super().__init__()
        self.enc_in = enc_in
        self.d_model = d_model
        self.patch_len = patch_len
        self.stride = stride

        if augmentations is None:
            augmentations = ["none"]

        self.register_buffer("coords", channel_positions.float(), persistent=False)
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        nn.init.xavier_uniform_(self.value_embedding.weight)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1024, d_model) * 0.02)
        self.channel_embedding = Electrode3DEmbedding(d_model)
        self.augmentation = nn.ModuleList([get_augmentation(aug, patch_len) for aug in augmentations])
        self.sr_embedding = nn.Embedding(NUM_SAMPLING_RATES, d_model)
        nn.init.zeros_(self.sr_embedding.weight)
        self.dropout = nn.Dropout(dropout)

    def _pad_to_stride(self, x):
        time_len = x.size(-1)
        if time_len < self.patch_len:
            pad_right = self.patch_len - time_len
        else:
            remainder = (time_len - self.patch_len) % self.stride
            pad_right = 0 if remainder == 0 else (self.stride - remainder)
        return F.pad(x, (0, pad_right), mode="replicate")

    def _sampling_rate_ids(self, fs, device):
        if fs is None:
            return None
        if not isinstance(fs, torch.Tensor):
            fs = torch.tensor(fs, device=device)
        fs = fs.to(device=device).view(-1)
        ids = []
        for value in fs.tolist():
            key = int(value)
            if key not in SAMPLING_RATE_TO_ID:
                raise ValueError(
                    f"LEAD only supports sampling-rate tokens {sorted(SAMPLING_RATE_TO_ID)}, got {key}."
                )
            ids.append(SAMPLING_RATE_TO_ID[key])
        return torch.tensor(ids, dtype=torch.long, device=device)

    def forward(self, x, fs=None):
        x = x.permute(0, 2, 1).contiguous()

        if self.training and len(self.augmentation) > 0:
            aug_idx = torch.randint(0, len(self.augmentation), (1,), device=x.device).item()
            x = self.augmentation[aug_idx](x)

        x = self._pad_to_stride(x)
        x = x.unfold(-1, self.patch_len, self.stride)
        batch_size, num_channels, num_patches, _ = x.shape
        x = rearrange(x, "b c n l -> (b c) n l")
        x = self.value_embedding(x)
        x = x + self.pos_embedding[:, :num_patches, :]
        x = rearrange(x, "(b c) n d -> b c n d", b=batch_size, c=num_channels)

        channel_embed = self.channel_embedding(self.coords.to(device=x.device, dtype=x.dtype))
        channel_embed = channel_embed.view(1, num_channels, 1, self.d_model)
        x = x + channel_embed
        x = rearrange(x, "b c n d -> b (c n) d")

        fs_ids = self._sampling_rate_ids(fs, x.device)
        if fs_ids is not None:
            sr_embed = self.sr_embedding(fs_ids).unsqueeze(1)
            x = x + sr_embed

        return self.dropout(x)
