import torch
import torch.nn as nn
from einops import rearrange

from ..modules.mixer_seq_simple import MixerModel


class EEGMamba(nn.Module):
    """Upstream EEGMamba backbone kept in a framework-friendly form."""

    def __init__(
        self,
        patch_size=200,
        d_model=200,
        n_layer=12,
        ssm_headdim=50,
        ssm_d_state=64,
    ):
        super().__init__()
        self.patch_embedding = PatchEmbedding(patch_size=patch_size, d_model=d_model)
        self.encoder = MixerModel(
            d_model=d_model,
            n_layer=n_layer,
            d_intermediate=0,
            ssm_cfg={
                "layer": "Mamba2",
                "headdim": ssm_headdim,
                "d_state": ssm_d_state,
            },
            attn_layer_idx=[],
            attn_cfg={},
            rms_norm=True,
            fused_add_norm=True,
            residual_in_fp32=True,
        )
        self.proj_out = nn.Sequential(nn.Linear(d_model, d_model))
        self.apply(_weights_init)

    def forward(self, x, mask=None):
        _, _, patch_num, _ = x.shape
        hidden_states = self.patch_embedding(x, mask=mask)
        hidden_states = rearrange(hidden_states, "b c s d -> b (c s) d")
        hidden_states = self.encoder(hidden_states)
        hidden_states = rearrange(hidden_states, "b (c s) d -> b c s d", s=patch_num)
        return self.proj_out(hidden_states)


class PatchEmbedding(nn.Module):
    """Patch embedding used by EEGMamba for patched EEG inputs."""

    def __init__(self, patch_size=200, d_model=200):
        super().__init__()
        if patch_size != 200:
            raise ValueError("EEGMamba currently requires patch_size=200.")
        if d_model != 200:
            raise ValueError("EEGMamba currently requires d_model=200 to match the pretrained architecture.")

        self.patch_size = patch_size
        self.d_model = d_model
        self.mask_encoding = nn.Parameter(torch.zeros(patch_size), requires_grad=False)
        self.positional_encoding = nn.Sequential(
            nn.Conv2d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=(7, 7),
                stride=(1, 1),
                padding=(3, 3),
                groups=d_model,
                bias=False,
            ),
        )
        self.proj_in = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=25,
                kernel_size=(1, 49),
                stride=(1, 25),
                padding=(0, 24),
                bias=False,
            ),
            nn.GroupNorm(5, 25),
            nn.GELU(),
        )
        self.spectral_proj = nn.Sequential(
            nn.Linear(patch_size // 2 + 1, d_model, bias=False),
            nn.Dropout(0.1),
        )

    def forward(self, x, mask=None):
        batch_size, num_channels, patch_num, patch_size = x.shape
        if patch_size != self.patch_size:
            raise ValueError(
                f"EEGMamba expects patched inputs with patch_size={self.patch_size}, got {patch_size}."
            )

        if mask is None:
            masked_x = x
        else:
            masked_x = x.clone()
            masked_x[mask == 1] = self.mask_encoding

        masked_x = rearrange(masked_x, "b c s p -> b p c s")
        time_x = rearrange(masked_x, "b p c s -> b (c s) p").unsqueeze(1)
        time_emb = self.proj_in(time_x)
        time_emb = time_emb.permute(0, 2, 1, 3).contiguous().view(
            batch_size, num_channels, patch_num, self.d_model
        )

        freq_x = rearrange(masked_x, "b p c s -> b c s p")
        spectral = torch.fft.rfft(freq_x, dim=-1, norm="forward")
        spectral = torch.abs(spectral)
        spectral_emb = self.spectral_proj(spectral)

        patch_emb = time_emb + spectral_emb
        positional_emb = self.positional_encoding(patch_emb.permute(0, 3, 1, 2))
        positional_emb = positional_emb.permute(0, 2, 3, 1)
        return patch_emb + positional_emb


def _weights_init(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(module, nn.Conv1d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(module, nn.BatchNorm1d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
