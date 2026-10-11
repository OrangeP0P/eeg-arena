"""HEAR fine-tuning wrapper for the unified EEG framework."""
import os
from functools import partial

import torch
import torch.nn as nn

from data.channels import MODEL_CHANNEL_SPACE_HEAR
from ..config import HEARConfig
from .backbone import NeuralTransformer
from .channel_info import LOCATION_STANDARD_1020


class Model(nn.Module):
    """Unified HEAR model wrapper. ``forward`` returns logits."""

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.patch_size = int(getattr(params, "patch_size", 200))
        self.variant = getattr(params, "variant", None) or HEARConfig.DEFAULT_VARIANT
        if self.variant not in HEARConfig.WEIGHT_VARIANTS:
            available = ", ".join(HEARConfig.WEIGHT_VARIANTS.keys())
            raise ValueError(f"Unknown HEAR variant '{self.variant}'. Available: {available}")
        variant_config = HEARConfig.WEIGHT_VARIANTS[self.variant]

        num_classes = int(getattr(params, "num_of_classes", 2))
        num_channels = int(getattr(params, "num_channels", 64))
        patch_num = int(getattr(params, "patch_num", 4))
        embed_dim = int(getattr(params, "embed_dim", 200))
        self.channel_mode = getattr(params, "hear_channel_mode", "heterogeneous")
        if self.channel_mode not in {"heterogeneous", "standard_1020"}:
            raise ValueError(
                "HEAR hear_channel_mode must be 'heterogeneous' or 'standard_1020', "
                f"got {self.channel_mode!r}."
            )
        layer_scale = getattr(params, "layer_scale_init_value", None)
        if layer_scale is None:
            layer_scale = variant_config["layer_scale_init_value"]

        self.backbone = NeuralTransformer(
            EEG_size=max(1, patch_num) * self.patch_size,
            patch_size=self.patch_size,
            in_chans=1,
            out_chans=8,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=int(variant_config["depth"]),
            num_heads=int(variant_config["num_heads"]),
            mlp_ratio=float(getattr(params, "mlp_ratio", 4.0)),
            qkv_bias=bool(getattr(params, "qkv_bias", True)),
            qk_norm=partial(nn.LayerNorm, eps=1e-6),
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            drop_rate=float(getattr(params, "drop_rate", 0.0)),
            attn_drop_rate=float(getattr(params, "attn_drop_rate", 0.0)),
            drop_path_rate=float(getattr(params, "drop_path_rate", 0.1)),
            init_values=float(layer_scale),
            use_abs_pos_emb=bool(getattr(params, "use_abs_pos_emb", True)),
            use_rel_pos_bias=bool(getattr(params, "use_rel_pos_bias", True)),
            use_mean_pooling=bool(getattr(params, "use_mean_pooling", True)),
            init_scale=float(getattr(params, "init_scale", 0.001)),
        )
        self.backbone.record_attn = False
        self.backbone.head = self._build_classifier(
            embed_dim=embed_dim,
            num_classes=num_classes,
            dropout=float(getattr(params, "dropout", 0.5)),
        )
        self.input_chans = None
        self.input_coords = None
        if self.channel_mode == "standard_1020":
            self.input_chans = self._resolve_input_chans(params, num_channels)
        else:
            self.input_coords = self._resolve_input_coords(params, num_channels)

        if getattr(params, "use_pretrained_weights", False):
            foundation_dir = getattr(
                params,
                "foundation_dir",
                "./models/Weights/HEAR/base/checkpoint.pth",
            )
            pretrained_file = self._resolve_pretrained_path(foundation_dir, self.variant)
            self._load_pretrained_weights(pretrained_file)

        if getattr(params, "frozen", False):
            self.freeze_backbone()

    def _build_classifier(self, embed_dim, num_classes, dropout):
        return nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.ELU(),
            nn.Linear(embed_dim, num_classes),
        )

    def _project_root(self):
        return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    def _resolve_pretrained_path(self, foundation_dir, variant):
        if not os.path.isabs(foundation_dir):
            foundation_dir = os.path.join(self._project_root(), foundation_dir)

        if os.path.isfile(foundation_dir):
            return foundation_dir

        variant_file = HEARConfig.WEIGHT_VARIANTS[variant]["file"]
        candidates = []
        if os.path.isdir(foundation_dir):
            candidates.extend(
                [
                    os.path.join(foundation_dir, os.path.basename(variant_file)),
                    os.path.join(foundation_dir, variant_file),
                    os.path.join(foundation_dir, "checkpoint.pth"),
                    os.path.join(foundation_dir, "checkpoint-49.pth"),
                ]
            )
        candidates.append(os.path.join(self._project_root(), HEARConfig.WEIGHTS_DIR, variant_file))

        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return candidates[0]

    def _extract_state_dict(self, checkpoint):
        if isinstance(checkpoint, dict):
            for key in ("model", "state_dict", "model_state_dict"):
                value = checkpoint.get(key)
                if isinstance(value, dict):
                    return value
        return checkpoint

    def _normalize_key(self, key):
        for prefix in ("module.", "backbone.", "student."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        return key

    def _load_pretrained_weights(self, pretrained_file):
        if not os.path.exists(pretrained_file):
            print(f"[HEAR] Warning: pretrained weights not found at {pretrained_file}")
            return

        checkpoint = torch.load(pretrained_file, map_location="cpu", weights_only=False)
        state_dict = self._extract_state_dict(checkpoint)
        target_state = self.backbone.state_dict()
        matched = {}
        skipped_shape = []
        unexpected = []

        for key, value in state_dict.items():
            if not isinstance(value, torch.Tensor):
                continue
            clean_key = self._normalize_key(key)
            if clean_key not in target_state:
                unexpected.append(clean_key)
                continue
            if target_state[clean_key].shape != value.shape:
                skipped_shape.append(clean_key)
                continue
            matched[clean_key] = value

        msg = self.backbone.load_state_dict(matched, strict=False)
        print(f"[HEAR] Loaded {self.variant} pretrained weights from {pretrained_file}")
        print(f"[HEAR] Matched {len(matched)}/{len(state_dict)} checkpoint keys")
        if skipped_shape:
            print(f"[HEAR] Skipped shape mismatches: {skipped_shape[:5]}... (total {len(skipped_shape)})")
        if unexpected:
            print(f"[HEAR] Unexpected checkpoint keys: {unexpected[:5]}... (total {len(unexpected)})")
        if msg.missing_keys:
            print(f"[HEAR] Missing keys: {msg.missing_keys[:5]}... (total {len(msg.missing_keys)})")
        if msg.unexpected_keys:
            print(f"[HEAR] Unexpected load keys: {msg.unexpected_keys[:5]}... (total {len(msg.unexpected_keys)})")

    def _resolve_input_chans(self, params, num_channels):
        model_channel_indices = getattr(params, "model_channel_indices", None)
        if isinstance(model_channel_indices, torch.Tensor):
            model_channel_indices = model_channel_indices.tolist()
        if model_channel_indices is not None:
            indices = [int(idx) for idx in model_channel_indices]
            if len(indices) == num_channels and all(0 <= idx < len(MODEL_CHANNEL_SPACE_HEAR) for idx in indices):
                return [0] + indices

        ch_names = getattr(params, "ch_names", None)
        if ch_names is not None:
            lookup = {
                self._channel_lookup_key(name): idx
                for idx, name in enumerate(MODEL_CHANNEL_SPACE_HEAR)
            }
            indices = []
            missing = []
            for name in list(ch_names)[:num_channels]:
                idx = self._resolve_channel_index(name, lookup)
                if idx is None:
                    missing.append(str(name))
                    continue
                indices.append(idx)
            if missing:
                raise ValueError(
                    "HEAR channel mapping failed for channels: "
                    + ", ".join(missing[:5])
                    + (" ..." if len(missing) > 5 else "")
                )
            if len(indices) == num_channels:
                return [0] + indices

        if num_channels > len(MODEL_CHANNEL_SPACE_HEAR):
            raise ValueError(
                f"HEAR received {num_channels} channels, but its channel space has "
                f"{len(MODEL_CHANNEL_SPACE_HEAR)} entries."
            )
        return [0] + list(range(num_channels))

    def _resolve_input_coords(self, params, num_channels):
        ch_names = list(getattr(params, "ch_names", []) or [])
        ch_positions = getattr(params, "ch_positions", None)
        if not ch_names and isinstance(ch_positions, dict):
            ch_names = list(ch_positions.keys())

        position_lookup = self._build_position_lookup(ch_positions)
        standard_lookup = {
            self._channel_lookup_key(name): [float(v) for v in coord[:3]]
            for name, coord in zip(MODEL_CHANNEL_SPACE_HEAR, LOCATION_STANDARD_1020)
        }

        coords = []
        missing = []
        for idx in range(num_channels):
            name = ch_names[idx] if idx < len(ch_names) else None
            coord = self._resolve_coord_for_channel(name, position_lookup, standard_lookup)
            if coord is None:
                missing.append(str(name if name is not None else idx))
                continue
            coords.append(coord)

        if missing:
            raise ValueError(
                "HEAR heterogeneous channel mode requires coordinates for every input channel. "
                "Missing coordinates for: "
                + ", ".join(missing[:5])
                + (" ..." if len(missing) > 5 else "")
            )
        return torch.tensor(coords, dtype=torch.float32)

    @staticmethod
    def _channel_lookup_key(name):
        text = str(name).strip()
        if ':' in text:
            text = text.split(':', 1)[1]
        elif ' ' in text:
            text = text.split()[-1]
        return text.replace(" ", "").upper()

    @classmethod
    def _resolve_channel_index(cls, name, lookup):
        key = cls._channel_lookup_key(name)
        if key in lookup:
            return lookup[key]
        if '-' in key:
            first_endpoint = key.split('-', 1)[0]
            return lookup.get(first_endpoint)
        return None

    @classmethod
    def _resolve_coord_for_channel(cls, name, position_lookup, standard_lookup):
        if name is None:
            return None
        key = cls._channel_lookup_key(name)
        lookup_key = key.split('-', 1)[0] if '-' in key else key
        coord = position_lookup.get(lookup_key)
        if coord is not None:
            return coord
        return standard_lookup.get(lookup_key)

    @classmethod
    def _build_position_lookup(cls, ch_positions):
        if not isinstance(ch_positions, dict):
            return {}
        lookup = {}
        for name, raw in ch_positions.items():
            coord = cls._normalize_coord(raw)
            if coord is None:
                continue
            key = cls._channel_lookup_key(name)
            lookup[key] = coord
        return lookup

    @staticmethod
    def _normalize_coord(raw):
        if raw is None:
            return None
        values = list(raw)
        if len(values) < 3:
            values = values + [0.0] * (3 - len(values))
        coord = [float(values[0]), float(values[1]), float(values[2])]
        if coord == [0.0, 0.0, 0.0]:
            return None
        return coord

    def extract_features(self, x):
        return self.backbone.forward_features(
            x,
            input_chans=self.input_chans,
            input_coords=self.input_coords,
        )

    def forward(self, x):
        return self.backbone(
            x,
            input_chans=self.input_chans,
            input_coords=self.input_coords,
        )

    def freeze_backbone(self):
        for name, param in self.backbone.named_parameters():
            if "head" not in name:
                param.requires_grad = False
        print("[HEAR] Backbone frozen (except classification head)")

    def get_param_groups(self):
        backbone_params = []
        head_params = []
        for name, param in self.backbone.named_parameters():
            if "head" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

        return [
            {"params": backbone_params, "lr_scale": getattr(self.params, "encoder_lr_scale", 1.0)},
            {"params": head_params, "lr_scale": getattr(self.params, "classifier_lr_scale", 10.0)},
        ]
