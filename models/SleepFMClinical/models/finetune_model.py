"""EEG-Arena adapter for all publicly released SleepFM Clinical transfer stacks."""

from __future__ import annotations

import os

import torch
from einops import rearrange
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F

from .upstream.models import (
    DiagnosisFinetuneFullLSTMCOXPHWithDemo,
    SetTransformer,
    SleepEventLSTMClassifier,
)


class Model(nn.Module):
    """SleepFM base encoder, optional official transfer stack, dynamic head."""

    PATCH_SIZE = 640  # five seconds at 128 Hz
    VARIANT_FILES = {
        "base-upstream": "model_base_best.pt",
        "base-safetensors": "model.safetensors",
        "sleep-staging": "model_sleep_staging_best.pth",
        "diagnosis": "model_diagnosis_best.pth",
    }

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.variant = str(getattr(params, "variant", None) or "base-upstream")
        use_pretrained = bool(getattr(params, "use_pretrained_weights", False))
        num_classes = int(getattr(params, "num_of_classes", 2))
        dropout = float(getattr(params, "classifier_dropout", 0.1))
        requested_path = str(
            getattr(
                params,
                "foundation_dir",
                "./models/Weights/SleepFM",
            )
        )
        weights_root = (
            requested_path
            if os.path.isdir(requested_path)
            else os.path.dirname(requested_path)
        )
        if not weights_root:
            weights_root = "."
        expected_file = self.VARIANT_FILES[self.variant]
        selected_path = (
            os.path.join(weights_root, expected_file)
            if os.path.isdir(requested_path)
            else requested_path
        )
        # The CLI default is a directory, but also make `--variant` robust when
        # an older config supplies the base file as its default foundation path.
        if os.path.basename(selected_path) != expected_file:
            variant_candidate = os.path.join(weights_root, expected_file)
            if os.path.isfile(variant_candidate):
                selected_path = variant_candidate

        self.backbone = SetTransformer(
            in_channels=1,
            patch_size=self.PATCH_SIZE,
            embed_dim=128,
            num_heads=8,
            num_layers=6,
            pooling_head=8,
            dropout=0.3,
            max_seq_length=128,
        )
        base_report = {
            "matched": 0,
            "available": 0,
            "expected": len(self.backbone.state_dict()),
            "missing": 0,
        }
        if use_pretrained:
            if self.variant == "base-safetensors":
                base_path = selected_path
            elif self.variant == "base-upstream":
                base_path = selected_path
            else:
                base_path = os.path.join(weights_root, "model_base_best.pt")
            base_report = self._load_module(
                self.backbone, base_path, require_fraction=0.95
            )

        self.transfer = None
        transfer_report = None
        if self.variant == "sleep-staging":
            self.transfer = SleepEventLSTMClassifier(
                embed_dim=128,
                num_heads=4,
                num_layers=1,
                num_classes=5,
                pooling_head=4,
                dropout=0.3,
                max_seq_length=8196,
            )
            transfer_report = {
                "matched": 0,
                "available": 0,
                "expected": len(self.transfer.state_dict()),
                "missing": 0,
            }
            if use_pretrained:
                transfer_report = self._load_module(
                    self.transfer, selected_path, require_fraction=0.95
                )
        elif self.variant == "diagnosis":
            self.transfer = DiagnosisFinetuneFullLSTMCOXPHWithDemo(
                embed_dim=128,
                num_heads=8,
                num_layers=2,
                num_classes=1065,
                pooling_head=8,
                dropout=0.3,
                max_seq_length=6480,
            )
            transfer_report = {
                "matched": 0,
                "available": 0,
                "expected": len(self.transfer.state_dict()),
                "missing": 0,
            }
            if use_pretrained:
                transfer_report = self._load_module(
                    self.transfer, selected_path, require_fraction=0.95
                )

        self.classifier = nn.Sequential(
            nn.LayerNorm(128), nn.Dropout(dropout), nn.Linear(128, num_classes)
        )
        self.pretrained_load_report = {
            "base": base_report,
            "transfer": transfer_report,
        }
        if getattr(params, "frozen", False):
            self.freeze_backbone()

    @staticmethod
    def _read_state(path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"SleepFM checkpoint not found: {path}")
        if path.endswith(".safetensors"):
            state = load_file(path, device="cpu")
        else:
            payload = torch.load(
                path, map_location="cpu", weights_only=False, mmap=True
            )
            state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        return {key.removeprefix("module."): value for key, value in state.items()}

    @classmethod
    def _load_module(cls, module: nn.Module, path: str, require_fraction: float):
        state = cls._read_state(path)
        expected = module.state_dict()
        # Braindecode's stable conversion only renames the positional buffer.
        if "positional_encoding" in state and "positional_encoding.pe" not in state:
            state["positional_encoding.pe"] = state.pop("positional_encoding")
        matched = {
            key: value
            for key, value in state.items()
            if key in expected and expected[key].shape == value.shape
        }
        fraction = len(matched) / max(1, len(expected))
        if fraction < require_fraction:
            raise RuntimeError(
                f"SleepFM checkpoint compatibility {fraction:.1%} is below "
                f"required {require_fraction:.1%}"
            )
        if len(matched) != len(expected) or len(matched) != len(state):
            raise RuntimeError(
                "SleepFM pinned checkpoint is not an exact state-dict match: "
                f"matched={len(matched)}, expected={len(expected)}, "
                f"available={len(state)}"
            )
        result = module.load_state_dict(matched, strict=True)
        print(
            f"[SleepFM Clinical] Strictly loaded {len(matched)}/{len(expected)} "
            f"tensors from {path}"
        )
        return {
            "matched": len(matched),
            "available": len(state),
            "expected": len(expected),
            "missing": len(result.missing_keys),
        }

    @staticmethod
    def _prepare_time(x: torch.Tensor):
        target = max(
            Model.PATCH_SIZE,
            min(
                Model.PATCH_SIZE * 128,
                ((x.shape[-1] + Model.PATCH_SIZE - 1) // Model.PATCH_SIZE)
                * Model.PATCH_SIZE,
            ),
        )
        if x.shape[-1] < target:
            x = F.pad(x, (0, target - x.shape[-1]))
        return x[..., :target]

    def _base_features(self, x: torch.Tensor):
        x = self._prepare_time(x)
        channel_mask = torch.zeros(
            x.shape[0], x.shape[1], dtype=torch.bool, device=x.device
        )
        pooled, temporal = self.backbone(x, channel_mask)
        return pooled, temporal

    def _transfer_features(self, temporal: torch.Tensor):
        # The official downstream checkpoints consume one embedding stream per
        # modality. EEG-only adaptation therefore supplies one present BAS stream.
        values = temporal.unsqueeze(1)  # [B, modality=1, S, 128]
        batch, modalities, steps, dim = values.shape
        mask = torch.zeros(batch, modalities, steps, dtype=torch.bool, device=values.device)
        values = rearrange(values, "b c s e -> (b s) c e")
        spatial_mask = rearrange(mask[:, :, 0].unsqueeze(1).expand(-1, steps, -1), "b t c -> (b t) c")
        values = self.transfer.spatial_pooling(values, spatial_mask)
        values = values.view(batch, steps, dim)
        if self.variant == "sleep-staging":
            values = self.transfer.positional_encoding(values)
            values = self.transfer.layer_norm(values)
            values = self.transfer.transformer_encoder(values)
        values, _ = self.transfer.lstm(values)
        return values.mean(dim=1)

    def extract_features(self, x: torch.Tensor):
        if x.ndim == 4:
            x = x.flatten(2)
        if x.ndim != 3:
            raise ValueError(f"SleepFM Clinical expects [B,C,T], got {tuple(x.shape)}")
        pooled, temporal = self._base_features(x)
        if self.transfer is None:
            return pooled
        return self._transfer_features(temporal)

    def forward(self, x: torch.Tensor):
        return self.classifier(self.extract_features(x))

    def freeze_backbone(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        if self.transfer is not None:
            for parameter in self.transfer.parameters():
                parameter.requires_grad = False

    def get_param_groups(self):
        encoder = list(self.backbone.parameters())
        if self.transfer is not None:
            encoder += list(self.transfer.parameters())
        return [
            {
                "params": [p for p in encoder if p.requires_grad],
                "lr_scale": float(getattr(self.params, "encoder_lr_scale", 1.0)),
            },
            {
                "params": [p for p in self.classifier.parameters() if p.requires_grad],
                "lr_scale": float(getattr(self.params, "classifier_lr_scale", 10.0)),
            },
        ]
