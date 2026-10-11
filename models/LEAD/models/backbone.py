"""Minimal LEADv2 backbone kept for unified fine-tuning."""

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .attention import LEADLayerV2
from .embedding import LEADv2Embedding, build_ordered_channel_positions
from .encoder import EncoderLayerV2, EncoderV2


class LEADv2Backbone(nn.Module):
    """Feature extractor adapted from the upstream LEADv2 model."""

    def __init__(self, params):
        super().__init__()
        self.enc_in = int(getattr(params, "num_channels", 64))
        self.d_model = int(getattr(params, "d_model", 128))
        self.patch_len = int(getattr(params, "patch_len", 50))
        self.stride = int(getattr(params, "stride", 50))
        self.dropout_rate = float(getattr(params, "dropout", 0.1))
        self.activation = getattr(params, "activation", "gelu")
        self.sampling_rate = int(getattr(params, "sampling_rate", 200))
        self.patch_num = 1

        ch_names = list(getattr(params, "ch_names", []) or [])
        ch_positions = getattr(params, "ch_positions", None)
        channel_positions = build_ordered_channel_positions(ch_names, ch_positions, self.enc_in)

        augmentations = getattr(params, "augmentations", "none")
        if isinstance(augmentations, str):
            augmentations = [item.strip() for item in augmentations.split(",") if item.strip()]
        if not augmentations:
            augmentations = ["none"]

        self.enc_embedding = LEADv2Embedding(
            enc_in=self.enc_in,
            d_model=self.d_model,
            patch_len=self.patch_len,
            stride=self.stride,
            dropout=self.dropout_rate,
            channel_positions=channel_positions,
            augmentations=augmentations,
        )
        self.encoder = EncoderV2(
            [
                EncoderLayerV2(
                    LEADLayerV2(
                        self.d_model,
                        self.enc_in,
                        int(getattr(params, "n_heads", 8)),
                        self.patch_num,
                        self.dropout_rate,
                        False,
                    ),
                    self.d_model,
                    int(getattr(params, "d_ff", 256)),
                    dropout=self.dropout_rate,
                    activation=self.activation,
                )
                for _ in range(int(getattr(params, "e_layers", 12)))
            ],
            norm_layer=nn.LayerNorm(self.d_model),
        )
        self.dropout = nn.Dropout(self.dropout_rate)

    @property
    def feature_dim(self):
        return self.enc_in * self.d_model

    def extract_features(self, x, fs=None):
        if x.ndim != 3:
            raise ValueError(f"LEAD expects [B, T, C] input, got shape {tuple(x.shape)}")
        if x.size(-1) != self.enc_in:
            raise ValueError(
                f"LEAD was initialized for {self.enc_in} channels, but received input with {x.size(-1)} channels."
            )

        enc_out = self.enc_embedding(x, fs)
        enc_out, _, _ = self.encoder(enc_out, attn_mask=None)
        batch_size, total_tokens, hidden_dim = enc_out.shape
        num_patches = total_tokens // self.enc_in
        enc_out = rearrange(enc_out, "b (c n) d -> b c n d", b=batch_size, c=self.enc_in, n=num_patches)
        enc_out = enc_out[:, :, -1, :]
        enc_out = F.gelu(enc_out)
        enc_out = self.dropout(enc_out)
        return enc_out.reshape(batch_size, self.enc_in * hidden_dim)
