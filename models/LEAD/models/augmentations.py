"""Minimal augmentation set required by LEADv2."""

import torch
import torch.nn as nn


class Jitter(nn.Module):
    def __init__(self, scale=0.1):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        if self.training:
            x = x + torch.randn_like(x) * self.scale
        return x


class Flip(nn.Module):
    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        if self.training and torch.rand(1, device=x.device).item() < self.prob:
            return torch.flip(x, dims=[-1])
        return x


class Shuffle(nn.Module):
    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        if self.training and torch.rand(1, device=x.device).item() < self.prob:
            return x[:, torch.randperm(x.size(1), device=x.device), :]
        return x


class TemporalMask(nn.Module):
    def __init__(self, ratio=0.1):
        super().__init__()
        self.ratio = ratio

    def forward(self, x):
        if not self.training:
            return x
        num_mask = int(x.size(-1) * self.ratio)
        if num_mask <= 0:
            return x
        masked = x.clone()
        mask_indices = torch.randperm(x.size(-1), device=x.device)[:num_mask]
        masked[:, :, mask_indices] = 0
        return masked


class ChannelMask(nn.Module):
    def __init__(self, ratio=0.1):
        super().__init__()
        self.ratio = ratio

    def forward(self, x):
        if not self.training:
            return x
        num_mask = int(x.size(1) * self.ratio)
        if num_mask <= 0:
            return x
        masked = x.clone()
        mask_indices = torch.randperm(x.size(1), device=x.device)[:num_mask]
        masked[:, mask_indices, :] = 0
        return masked


class PatchMask(nn.Module):
    def __init__(self, patch_len=25, ratio=0.1):
        super().__init__()
        self.patch_len = patch_len
        self.ratio = ratio

    def forward(self, x):
        if not self.training:
            return x
        batch, channels, time_len = x.shape
        num_patches = time_len // self.patch_len
        total_units = channels * num_patches
        num_mask = int(total_units * self.ratio)
        if num_mask <= 0 or total_units == 0:
            return x
        masked = x.clone()
        indices = torch.randperm(total_units, device=x.device)[:num_mask]
        channel_indices = indices // num_patches
        patch_indices = indices % num_patches
        for channel_index, patch_index in zip(channel_indices.tolist(), patch_indices.tolist()):
            start = patch_index * self.patch_len
            end = start + self.patch_len
            masked[:, channel_index, start:end] = 0
        return masked


class FrequencyMask(nn.Module):
    def __init__(self, ratio=0.1):
        super().__init__()
        self.ratio = ratio

    def forward(self, x):
        if not self.training:
            return x
        x_fft = torch.fft.rfft(x, dim=-1)
        mask = torch.rand(x_fft.shape, device=x.device) > self.ratio
        x_fft = x_fft * mask
        return torch.fft.irfft(x_fft, n=x.size(-1), dim=-1)


def get_augmentation(augmentation, patch_len=25):
    if augmentation.startswith("jitter"):
        return Jitter() if len(augmentation) == 6 else Jitter(float(augmentation[6:]))
    if augmentation.startswith("drop"):
        return nn.Dropout(0.1) if len(augmentation) == 4 else nn.Dropout(float(augmentation[4:]))
    if augmentation.startswith("flip"):
        return Flip() if len(augmentation) == 4 else Flip(float(augmentation[4:]))
    if augmentation.startswith("shuffle"):
        return Shuffle() if len(augmentation) == 7 else Shuffle(float(augmentation[7:]))
    if augmentation.startswith("frequency"):
        return FrequencyMask() if len(augmentation) == 9 else FrequencyMask(float(augmentation[9:]))
    if augmentation.startswith("mask"):
        return TemporalMask() if len(augmentation) == 4 else TemporalMask(float(augmentation[4:]))
    if augmentation.startswith("channel"):
        return ChannelMask() if len(augmentation) == 7 else ChannelMask(float(augmentation[7:]))
    if augmentation.startswith("patch"):
        return PatchMask(patch_len) if len(augmentation) == 5 else PatchMask(patch_len, float(augmentation[5:]))
    if augmentation == "none":
        return nn.Identity()
    raise ValueError(f"Unknown augmentation {augmentation}")
