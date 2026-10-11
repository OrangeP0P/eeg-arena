"""STEEGFormer fine-tuning wrapper for the unified EEG framework."""
import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.channels import MODEL_CHANNEL_SPACE_STEEGFORMER

from ..config import STEEGFormerConfig
from .backbone import build_steegformer_backbone


def _canonical_channel_key(name):
    text = str(name).replace("EEG:", "").strip()
    return re.sub(r"[^A-Za-z0-9]", "", text).upper()


class Model(nn.Module):
    """Unified STEEGFormer wrapper where `forward(x)` returns logits."""

    PATCH_SIZE = 16
    MAX_TEMPORAL_PATCHES = 511

    def __init__(self, params):
        super().__init__()
        self.params = params

        self.channel_space = list(MODEL_CHANNEL_SPACE_STEEGFORMER)
        self.channel_lookup = {
            _canonical_channel_key(name): idx for idx, name in enumerate(self.channel_space)
        }

        variant = getattr(params, "variant", None) or STEEGFormerConfig.DEFAULT_VARIANT
        if variant not in STEEGFormerConfig.WEIGHT_VARIANTS:
            available = ", ".join(STEEGFormerConfig.WEIGHT_VARIANTS.keys())
            raise ValueError(f"Unknown STEEGFormer variant '{variant}'. Available: {available}")
        self.variant = variant
        variant_cfg = STEEGFormerConfig.WEIGHT_VARIANTS[variant]

        num_classes = int(getattr(params, "num_of_classes", 2))
        classifier_dropout = float(getattr(params, "classifier_dropout", 0.1))

        self.backbone = build_steegformer_backbone(
            embed_dim=int(variant_cfg["embed_dim"]),
            depth=int(variant_cfg["depth"]),
            num_heads=int(variant_cfg["num_heads"]),
            num_classes=num_classes,
            classifier_dropout=classifier_dropout,
            num_channel_embeddings=int(variant_cfg["num_channel_embeddings"]),
            global_pool=True,
            attn_drop_rate=0.1,
            proj_drop_rate=0.1,
            drop_path_rate=0.1,
        )

        if getattr(params, "use_pretrained_weights", False):
            foundation_dir = getattr(
                params,
                "foundation_dir",
                "./models/Weights/STEEGFormer",
            )
            pretrained_file = self._resolve_pretrained_path(foundation_dir, variant)
            self._load_pretrained_weights(pretrained_file)

        if getattr(params, "frozen", False):
            self.freeze_backbone()

    def _resolve_pretrained_path(self, foundation_dir, variant):
        if os.path.isfile(foundation_dir):
            return foundation_dir

        variant_file = STEEGFormerConfig.WEIGHT_VARIANTS[variant]["file"]
        variant_basename = os.path.basename(variant_file)

        if os.path.isdir(foundation_dir):
            direct_candidate = os.path.join(foundation_dir, variant_file)
            if os.path.exists(direct_candidate):
                return direct_candidate

            basename_candidate = os.path.join(foundation_dir, variant_basename)
            if os.path.exists(basename_candidate):
                return basename_candidate

            fallback_candidates = [
                variant_basename,
                "model.pth",
                "checkpoint.pth",
                "checkpoint-300.pth",
                "checkpoint-288.pth",
                "large_weights_only_196.pth",
                "large_weights_only_210.pth",
            ]
            for candidate_name in fallback_candidates:
                candidate = os.path.join(foundation_dir, candidate_name)
                if os.path.exists(candidate):
                    return candidate

            return direct_candidate

        return foundation_dir

    @staticmethod
    def _extract_state_dict(checkpoint):
        if not isinstance(checkpoint, dict):
            return checkpoint

        for key in ("model", "state_dict", "model_state_dict", "net"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        return checkpoint

    @staticmethod
    def _normalize_key(key):
        normalized = key
        for prefix in ("module.", "model.", "backbone.", "encoder."):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
        return normalized

    def _load_pretrained_weights(self, pretrained_file):
        if not os.path.exists(pretrained_file):
            print(f"[STEEGFormer] Warning: pretrained weights not found at {pretrained_file}")
            return

        checkpoint = torch.load(pretrained_file, map_location="cpu", weights_only=False)
        state_dict = self._extract_state_dict(checkpoint)
        normalized = {
            self._normalize_key(key): value
            for key, value in state_dict.items()
            if isinstance(value, torch.Tensor)
        }

        model_state = self.backbone.state_dict()
        matched = {
            key: value
            for key, value in normalized.items()
            if key in model_state and model_state[key].shape == value.shape
        }
        mismatched = [
            key
            for key, value in normalized.items()
            if key in model_state and model_state[key].shape != value.shape
        ]

        msg = self.backbone.load_state_dict(matched, strict=False)
        print(f"[STEEGFormer] Loaded pretrained weights from {pretrained_file}")
        print(f"[STEEGFormer] Matched {len(matched)}/{len(normalized)} keys")
        if mismatched:
            print(
                "[STEEGFormer] Skipped (shape mismatch): "
                f"{mismatched[:5]}... (total {len(mismatched)})"
            )
        if msg.missing_keys:
            print(
                "[STEEGFormer] Missing keys: "
                f"{msg.missing_keys[:5]}... (total {len(msg.missing_keys)})"
            )
        if msg.unexpected_keys:
            print(
                "[STEEGFormer] Unexpected keys: "
                f"{msg.unexpected_keys[:5]}... (total {len(msg.unexpected_keys)})"
            )

    def _prepare_input(self, x):
        if x.dim() == 4:
            batch, channels, steps, patch_size = x.shape
            x = x.reshape(batch, channels, steps * patch_size)
        elif x.dim() != 3:
            raise ValueError(f"STEEGFormer expects 3D or 4D EEG input, got shape {tuple(x.shape)}")

        time_len = x.shape[-1]
        pad_len = (self.PATCH_SIZE - (time_len % self.PATCH_SIZE)) % self.PATCH_SIZE
        if pad_len > 0:
            x = F.pad(x, (0, pad_len))

        num_patches = x.shape[-1] // self.PATCH_SIZE
        if num_patches > self.MAX_TEMPORAL_PATCHES:
            raise ValueError(
                "STEEGFormer received too many temporal patches: "
                f"{num_patches} > {self.MAX_TEMPORAL_PATCHES}. "
                "Reduce the raw window length before fine-tuning."
            )

        return x

    def _resolve_channel_indices(self, num_channels):
        model_channel_indices = getattr(self.params, "model_channel_indices", None)
        if isinstance(model_channel_indices, torch.Tensor):
            model_channel_indices = model_channel_indices.tolist()
        if model_channel_indices is not None:
            model_channel_indices = [int(idx) for idx in model_channel_indices]
            if len(model_channel_indices) == num_channels:
                return model_channel_indices

        ch_names = getattr(self.params, "ch_names", None)
        if ch_names is not None:
            mapped = []
            missing = []
            for name in list(ch_names)[:num_channels]:
                key = _canonical_channel_key(name)
                if key in self.channel_lookup:
                    mapped.append(self.channel_lookup[key])
                else:
                    missing.append(name)
            if missing:
                raise ValueError(
                    "STEEGFormer channel mapping failed for channels: "
                    + ", ".join(map(str, missing[:5]))
                    + (" ..." if len(missing) > 5 else "")
                )
            if len(mapped) == num_channels:
                return mapped

        if num_channels == len(self.channel_space):
            return list(range(len(self.channel_space)))

        raise ValueError(
            "STEEGFormer could not infer channel indices for the current EEG batch. "
            "Expected shared data-layer metadata `model_channel_indices` or mappable `ch_names`."
        )

    def _build_batch_channel_indices(self, x):
        batch_size, num_channels, _ = x.shape
        channel_indices = self._resolve_channel_indices(num_channels)
        channel_indices = torch.tensor(channel_indices, dtype=torch.long, device=x.device)
        return channel_indices.unsqueeze(0).expand(batch_size, -1)

    def extract_features(self, x):
        x = self._prepare_input(x)
        chan_idx = self._build_batch_channel_indices(x)
        return self.backbone.forward_features(x, chan_idx)

    def forward(self, x):
        x = self._prepare_input(x)
        chan_idx = self._build_batch_channel_indices(x)
        return self.backbone(x, chan_idx)

    def freeze_backbone(self):
        for name, param in self.backbone.named_parameters():
            if name.startswith("head."):
                continue
            param.requires_grad = False
        print("[STEEGFormer] Backbone frozen (only classification head remains trainable)")

    def get_param_groups(self):
        encoder_lr_scale = float(getattr(self.params, "encoder_lr_scale", 1.0))
        classifier_lr_scale = float(getattr(self.params, "classifier_lr_scale", 10.0))

        backbone_params = []
        classifier_params = []
        for name, param in self.backbone.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("head.") or name.startswith("fc_norm."):
                classifier_params.append(param)
            else:
                backbone_params.append(param)

        return [
            {"params": backbone_params, "lr_scale": encoder_lr_scale},
            {"params": classifier_params, "lr_scale": classifier_lr_scale},
        ]
