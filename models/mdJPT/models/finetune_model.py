"""mdJPT fine-tuning wrapper for the unified EEG framework."""
import os
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import MdJPTBackbone


class Model(nn.Module):
    """Unified mdJPT model: `forward(x)` returns logits."""

    def __init__(self, params):
        super().__init__()
        self.params = params

        self.patch_size = int(getattr(params, "patch_size", 200))
        self.patch_num = int(getattr(params, "patch_num", 1))
        self.input_time_len = self.patch_size * self.patch_num

        self.num_channels = int(getattr(params, "num_channels", 64))
        self.num_classes = int(getattr(params, "num_of_classes", 2))

        self.feature_pooling = str(getattr(params, "feature_pooling", "mean")).lower()
        self.dropout = float(getattr(params, "dropout", 0.1))

        dilation_array = self._parse_dilation_array(getattr(params, "dilation_array", [1, 3, 6, 12]))

        self.backbone = MdJPTBackbone(
            num_input_channels=self.num_channels,
            input_time_len=self.input_time_len,
            patch_len=int(getattr(params, "patch_len", 32)),
            patch_stride=int(getattr(params, "patch_stride", 6)),
            d_model=int(getattr(params, "d_model", 32)),
            n_heads=int(getattr(params, "n_heads", 8)),
            n_layers=int(getattr(params, "n_layers", 3)),
            d_ff=int(getattr(params, "d_ff", 256)),
            transformer_dropout=float(getattr(params, "transformer_dropout", 0.0)),
            attn_dropout=float(getattr(params, "attn_dropout", 0.0)),
            target_channels=int(getattr(params, "target_channels", 64)),
            n_ms_filters=int(getattr(params, "n_ms_filters", 4)),
            ms_filter_time_len=int(getattr(params, "ms_filter_time_len", 3)),
            dilation_array=dilation_array,
            seg_att=int(getattr(params, "seg_att", 15)),
            has_att=bool(getattr(params, "has_att", True)),
            global_att=bool(getattr(params, "global_att", False)),
        )

        self.feature_dim = int(self.backbone.feature_dim)
        classifier_hidden_dim = int(getattr(params, "classifier_hidden_dim", 0))
        if classifier_hidden_dim <= 0:
            classifier_hidden_dim = self.feature_dim

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(classifier_hidden_dim, self.num_classes),
        )

        if getattr(params, "use_pretrained_weights", False):
            foundation_dir = getattr(
                params,
                "foundation_dir",
                "./models/Weights/mdJPT/mdjpt_pretrained.pth",
            )
            pretrained_file = self._resolve_pretrained_path(foundation_dir)
            self._load_pretrained_weights(pretrained_file)

        if getattr(params, "frozen", False):
            self.freeze_backbone()

    @staticmethod
    def _parse_dilation_array(value) -> tuple[int, int, int, int]:
        if isinstance(value, str):
            parts = [p.strip() for p in value.split(",") if p.strip()]
            arr = [int(p) for p in parts]
        elif isinstance(value, Iterable):
            arr = [int(v) for v in value]
        else:
            arr = [int(value)]

        if not arr:
            arr = [1, 3, 6, 12]

        if len(arr) < 4:
            arr = arr + [arr[-1]] * (4 - len(arr))
        return tuple(arr[:4])

    @staticmethod
    def _resolve_pretrained_path(foundation_dir: str) -> str:
        if os.path.isfile(foundation_dir):
            return foundation_dir

        if os.path.isdir(foundation_dir):
            for candidate in (
                "mdjpt_pretrained.pth",
                "mdJPT_pretrained.pth",
                "model.pth",
                "checkpoint.pth",
                "last.ckpt",
            ):
                candidate_path = os.path.join(foundation_dir, candidate)
                if os.path.exists(candidate_path):
                    return candidate_path
            return os.path.join(foundation_dir, "mdjpt_pretrained.pth")

        return foundation_dir

    @staticmethod
    def _extract_state_dict(checkpoint):
        if not isinstance(checkpoint, dict):
            return checkpoint

        for key in ("state_dict", "model", "net", "backbone"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value

        return checkpoint

    @staticmethod
    def _normalize_key(key: str) -> str:
        normalized = key

        # Common prefixes in DDP / Lightning checkpoints.
        for prefix in ("module.", "model.", "net.", "backbone."):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]

        # Upstream mdJPT naming from pretraining code.
        replacements = {
            "patchTST.": "patch_encoder.",
            "c_mlps.0.": "channel_projector.",
            "cnn_encoder.": "cnn_encoder.",
        }
        for src, dst in replacements.items():
            if normalized.startswith(src):
                normalized = dst + normalized[len(src):]

        return normalized

    def _load_pretrained_weights(self, pretrained_file: str):
        if not os.path.exists(pretrained_file):
            print(f"[mdJPT] Warning: pretrained weights not found at {pretrained_file}")
            return

        checkpoint = torch.load(pretrained_file, map_location="cpu", weights_only=False)
        state_dict = self._extract_state_dict(checkpoint)

        normalized = {}
        for key, value in state_dict.items():
            if not isinstance(value, torch.Tensor):
                continue
            normalized[self._normalize_key(key)] = value

        model_state = self.backbone.state_dict()
        matched = {
            k: v
            for k, v in normalized.items()
            if k in model_state and model_state[k].shape == v.shape
        }
        skipped_mismatch = [
            k
            for k, v in normalized.items()
            if k in model_state and model_state[k].shape != v.shape
        ]

        msg = self.backbone.load_state_dict(matched, strict=False)
        print(f"[mdJPT] Loaded pretrained weights from {pretrained_file}")
        print(f"[mdJPT] Matched {len(matched)}/{len(normalized)} keys")
        if skipped_mismatch:
            print(
                f"[mdJPT] Skipped (shape mismatch): {skipped_mismatch[:5]}... "
                f"(total {len(skipped_mismatch)})"
            )
        if msg.missing_keys:
            print(f"[mdJPT] Missing keys: {msg.missing_keys[:5]}... (total {len(msg.missing_keys)})")
        if msg.unexpected_keys:
            print(
                f"[mdJPT] Unexpected keys: {msg.unexpected_keys[:5]}... "
                f"(total {len(msg.unexpected_keys)})"
            )

    def _prepare_input(self, x):
        if x.dim() == 4:
            # Patched format: [B, C, S, P] -> [B, C, S*P]
            bsz, ch, steps, patch = x.shape
            x = x.reshape(bsz, ch, steps * patch)
        elif x.dim() != 3:
            raise ValueError(f"mdJPT expects 3D or 4D EEG input, got shape {tuple(x.shape)}")

        if x.shape[-1] < self.input_time_len:
            x = F.pad(x, (0, self.input_time_len - x.shape[-1]))
        elif x.shape[-1] > self.input_time_len:
            x = x[..., : self.input_time_len]

        return x

    def extract_features(self, x):
        x = self._prepare_input(x)
        fmap = self.backbone(x)  # [B, D, 1, T]

        if self.feature_pooling == "mean":
            return fmap.mean(dim=(2, 3))
        if self.feature_pooling == "max":
            return fmap.amax(dim=(2, 3))

        raise ValueError(
            f"Unsupported feature_pooling='{self.feature_pooling}'. Use 'mean' or 'max'."
        )

    def forward(self, x):
        features = self.extract_features(x)
        return self.classifier(features)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("[mdJPT] Backbone frozen")

    def get_param_groups(self):
        encoder_lr_scale = float(getattr(self.params, "encoder_lr_scale", 1.0))
        classifier_lr_scale = float(getattr(self.params, "classifier_lr_scale", 10.0))

        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        classifier_params = [p for p in self.classifier.parameters() if p.requires_grad]

        return [
            {"params": backbone_params, "lr_scale": encoder_lr_scale},
            {"params": classifier_params, "lr_scale": classifier_lr_scale},
        ]
