import copy
import torch
import torch.nn as nn
from timm.models.layers import DropPath


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None, drop_path_rate=0.0):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = copy.deepcopy(encoder_layer)
            layer.set_drop_path(dpr[i])
            self.layers.append(layer)
        self.norm = norm

    def forward(self, src):
        output = src
        for layer in self.layers:
            output = layer(output)
        if self.norm is not None:
            output = self.norm(output)
        return output


class DynamicFactorizedAttention(nn.Module):
    def __init__(self, d_model, nhead, group_size=4, drop_path=0.0):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.group_size = group_size
        self.temporal_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm_temporal = nn.LayerNorm(d_model)
        self.spatial_proj = nn.Linear(d_model, d_model * 3)
        self.spatial_out = nn.Linear(d_model, d_model)
        self.norm_spatial = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def set_drop_path(self, drop_path_prob):
        self.drop_path = (
            DropPath(drop_path_prob) if drop_path_prob > 0 else nn.Identity()
        )

    def forward(self, src):
        batch_size, channels, patches, dim = src.shape
        temporal_in = src.reshape(batch_size * channels, patches, dim)
        attn_out = self.temporal_attn(
            self.norm_temporal(temporal_in),
            temporal_in,
            temporal_in,
        )[0]
        temporal_out = temporal_in + self.drop_path(attn_out)
        spatial_in = temporal_out.view(batch_size, channels, patches, dim)
        spatial_features = []
        num_groups = (channels + self.group_size - 1) // self.group_size
        for group_idx in range(num_groups):
            start_idx = group_idx * self.group_size
            end_idx = min((group_idx + 1) * self.group_size, channels)
            group_channels = end_idx - start_idx
            if group_channels == 0:
                continue
            group_feat = (
                spatial_in[:, start_idx:end_idx, :, :]
                .permute(0, 2, 1, 3)
                .reshape(batch_size * patches, group_channels, dim)
            )
            q, k, v = self.spatial_proj(group_feat).chunk(3, dim=-1)
            attn_out, _ = self.temporal_attn(q, k, v)
            attn_out = (
                self.spatial_out(attn_out)
                .reshape(batch_size, patches, group_channels, dim)
                .permute(0, 2, 1, 3)
            )
            spatial_features.append(attn_out)
        spatial_calc = (
            torch.cat(spatial_features, dim=1) if spatial_features else spatial_in
        )
        spatial_out = self.norm_spatial(spatial_in + self.drop_path(spatial_calc))
        return spatial_out + self.drop_path(self.ffn(self.norm_ffn(spatial_out)))


class MultiScalePatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_size, max_channels=256, max_patches=100):
        super().__init__()
        self.d_model = d_model
        out_channels = 16
        self.scale_s = nn.Sequential(
            nn.Conv2d(
                1, out_channels, kernel_size=(1, 15), stride=(1, 5), padding=(0, 7)
            ),
            nn.GroupNorm(4, out_channels),
            nn.GELU(),
        )
        self.scale_m = nn.Sequential(
            nn.Conv2d(
                1, out_channels, kernel_size=(1, 49), stride=(1, 25), padding=(0, 24)
            ),
            nn.GroupNorm(4, out_channels),
            nn.GELU(),
        )
        self.scale_l = nn.Sequential(
            nn.Conv2d(
                1, out_channels, kernel_size=(1, 101), stride=(1, 50), padding=(0, 50)
            ),
            nn.GroupNorm(4, out_channels),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(20)
        self.proj_in = nn.Linear(out_channels * 3 * 20, d_model)
        self.channel_pos_emb = nn.Parameter(torch.randn(1, max_channels, 1, d_model))
        self.patch_pos_emb = nn.Parameter(torch.randn(1, 1, max_patches, d_model))

    def forward(self, x):
        batch_size, channels, patches, patch_size = x.shape
        x = x.view(batch_size, channels * patches, patch_size).unsqueeze(1)
        scale_outputs = [self.scale_s(x), self.scale_m(x), self.scale_l(x)]
        pooled = []
        for out in scale_outputs:
            out = out.permute(0, 2, 1, 3)
            out = out.reshape(
                batch_size * channels * patches, out.shape[2], out.shape[3]
            )
            out = self.pool(out).reshape(batch_size * channels * patches, -1)
            pooled.append(out)
        patch_emb = self.proj_in(torch.cat(pooled, dim=1))
        patch_emb = patch_emb.view(batch_size, channels, patches, self.d_model)
        channel_pos = self.channel_pos_emb[:, :channels, :, :]
        patch_pos = self.patch_pos_emb[:, :, :patches, :]
        return patch_emb + channel_pos + patch_pos


class UniDMFM(nn.Module):
    def __init__(
        self,
        num_layers=12,
        nhead=12,
        d_model=384,
        patch_size=200,
        projection_dim=256,
        stft_params=None,
        drop_path_rate=0.1,
    ):
        super().__init__()
        self.model_name = "Uni-DMFM"
        self.d_model = d_model
        self.patch_size = patch_size
        self.stft_params = stft_params or {"n_fft": 64, "hop_length": 16}
        self.patch_embedding = MultiScalePatchEmbedding(d_model, patch_size)
        encoder_layer = DynamicFactorizedAttention(d_model=d_model, nhead=nhead)
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            drop_path_rate=drop_path_rate,
        )
        self.time_reconstruction_head = nn.Linear(d_model, patch_size)
        stft_output_dim = self._calculate_stft_dim()
        self.freq_reconstruction_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, stft_output_dim),
        )
        self.contrastive_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, projection_dim),
        )

    def _calculate_stft_dim(self):
        n_fft = self.stft_params["n_fft"]
        hop_length = self.stft_params["hop_length"]
        dummy_input = torch.randn(1, self.patch_size)
        stft_out = torch.stft(
            dummy_input,
            n_fft=n_fft,
            hop_length=hop_length,
            return_complex=True,
            window=torch.hann_window(n_fft),
        )
        return stft_out.shape[1] * stft_out.shape[2]

    def forward_features(self, x):
        x = self.patch_embedding(x)
        return self.transformer_encoder(x)

    def forward(self, x, mode="full"):
        features = self.forward_features(x)
        if mode == "full":
            time_recon = self.time_reconstruction_head(features)
            freq_recon = self.freq_reconstruction_head(features)
            pooled_features = features.mean(dim=[1, 2])
            contrastive_proj = self.contrastive_head(pooled_features)
            return time_recon, freq_recon, contrastive_proj
        if mode == "contrastive":
            pooled_features = features.mean(dim=[1, 2])
            return self.contrastive_head(pooled_features)
        if mode == "finetune":
            return features
        raise ValueError(f"Unknown mode: {mode}")

