"""Frequency-domain patch embedder used by LUNA."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
from einops import rearrange
from timm.layers import Mlp


class FrequencyFeatureEmbedder(nn.Module):
    """Embed FFT magnitude/phase features for each temporal patch."""

    def __init__(self, patch_size, embed_dim):
        super().__init__()
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)
        in_features = 2 * (self.patch_size // 2 + 1)
        self.frequency_to_embed = Mlp(
            in_features=in_features,
            hidden_features=int(4 * in_features),
            out_features=self.embed_dim,
        )

    def forward(self, x):
        batch, channels, time_len = x.size()
        if time_len % self.patch_size != 0:
            pad_size = self.patch_size - (time_len % self.patch_size)
            x = F.pad(x, (0, pad_size))
            time_len = x.size(-1)

        steps = time_len // self.patch_size
        x = x.view(batch, channels, steps, self.patch_size)

        freq = fft.rfft(x, dim=-1)
        magnitude = torch.abs(freq)
        phase = torch.angle(freq)
        freq_features = torch.cat((magnitude, phase), dim=-1)

        embedded = self.frequency_to_embed(freq_features)
        embedded = rearrange(embedded, "B C S D -> B (C S) D")
        return embedded
