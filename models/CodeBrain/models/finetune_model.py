"""CodeBrain wrapper that adapts the upstream SSSM backbone to the unified interface."""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sssm import SSSM


class Model(nn.Module):
    """Unified CodeBrain fine-tuning model."""

    def __init__(self, params):
        super().__init__()
        self.params = params

        num_classes = int(getattr(params, "num_of_classes", 2))
        num_channels = int(getattr(params, "num_channels", 64))
        patch_num = int(getattr(params, "patch_num", 1))
        patch_size = int(getattr(params, "patch_size", 200))
        d_model = int(getattr(params, "d_model", 200))
        n_layer = int(getattr(params, "n_layer", 8))
        dropout = float(getattr(params, "dropout", 0.1))
        codebook_size_t = int(getattr(params, "codebook_size_t", 4096))
        codebook_size_f = int(getattr(params, "codebook_size_f", 4096))
        classifier_hidden = max(4, patch_num) * d_model

        self.num_channels = num_channels
        self.patch_num = patch_num
        self.feature_width = patch_size
        self.flatten_dim = num_channels * patch_num * patch_size
        self.is_binary_task = num_classes == 2
        output_dim = 1 if self.is_binary_task else num_classes

        if patch_size != 200:
            raise ValueError(
                f"CodeBrain requires patch_size=200, but got patch_size={patch_size}."
            )

        # Keep S4 kernel configuration aligned with upstream pretrained checkpoints.
        self.backbone = SSSM(
            in_channels=patch_size,
            res_channels=d_model,
            skip_channels=d_model,
            out_channels=patch_size,
            num_res_layers=n_layer,
            diffusion_step_embed_dim_in=d_model,
            diffusion_step_embed_dim_mid=d_model,
            diffusion_step_embed_dim_out=d_model,
            s4_lmax=570,
            s4_d_state=64,
            s4_dropout=dropout,
            s4_bidirectional=True,
            s4_layernorm=True,
            codebook_size_t=codebook_size_t,
            codebook_size_f=codebook_size_f,
            if_codebook=False,
        )

        if getattr(params, "use_pretrained_weights", False):
            foundation_dir = getattr(
                params,
                "foundation_dir",
                "./models/Weights/CodeBrain/CodeBrain.pth",
            )
            pretrained_file = self._resolve_pretrained_path(foundation_dir)
            self._load_pretrained_weights(pretrained_file)

        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_dim, classifier_hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, d_model),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim),
        )

        if getattr(params, "frozen", False):
            self.freeze_backbone()

    def _resolve_pretrained_path(self, foundation_dir):
        if os.path.isfile(foundation_dir):
            return foundation_dir

        if os.path.isdir(foundation_dir):
            for candidate in ("CodeBrain.pth", "codebrain.pth", "checkpoint.pth"):
                candidate_path = os.path.join(foundation_dir, candidate)
                if os.path.exists(candidate_path):
                    return candidate_path
            return os.path.join(foundation_dir, "CodeBrain.pth")

        return foundation_dir

    def _load_pretrained_weights(self, pretrained_file):
        if not os.path.exists(pretrained_file):
            print(f"[CodeBrain] Warning: pretrained weights not found at {pretrained_file}")
            return

        checkpoint = torch.load(pretrained_file, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        normalized_state_dict = {}
        for key, value in state_dict.items():
            clean_key = key
            for prefix in ("module.", "backbone."):
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix):]
            normalized_state_dict[clean_key] = value

        model_state_dict = self.backbone.state_dict()
        matched = {
            k: v for k, v in normalized_state_dict.items()
            if k in model_state_dict and model_state_dict[k].shape == v.shape
        }
        skipped_mismatch = [
            k for k, v in normalized_state_dict.items()
            if k in model_state_dict and model_state_dict[k].shape != v.shape
        ]

        msg = self.backbone.load_state_dict(matched, strict=False)
        print(f"[CodeBrain] Loaded pretrained weights from {pretrained_file}")
        print(f"[CodeBrain] Matched {len(matched)}/{len(normalized_state_dict)} keys")
        if skipped_mismatch:
            print(
                f"[CodeBrain] Skipped (shape mismatch): {skipped_mismatch[:5]}... "
                f"(total {len(skipped_mismatch)})"
            )
        if msg.missing_keys:
            print(f"[CodeBrain] Missing keys: {msg.missing_keys[:5]}... (total {len(msg.missing_keys)})")
        if msg.unexpected_keys:
            print(
                f"[CodeBrain] Unexpected keys: {msg.unexpected_keys[:5]}... "
                f"(total {len(msg.unexpected_keys)})"
            )

    def extract_features(self, x):
        """Return flattened backbone features in the original downstream style."""
        feats = self.backbone(x)
        if feats.dim() != 4:
            raise RuntimeError(f"CodeBrain backbone output must be 4D, got shape {tuple(feats.shape)}")
        batch_size, num_channels, patch_num, feature_width = feats.shape
        expected_shape = (self.num_channels, self.patch_num, self.feature_width)
        actual_shape = (num_channels, patch_num, feature_width)
        if actual_shape != expected_shape:
            raise RuntimeError(
                "CodeBrain feature shape does not match dataloader metadata: "
                f"expected {expected_shape}, got {actual_shape}."
            )
        return feats.contiguous().view(batch_size, self.flatten_dim)

    def forward(self, x):
        feats = self.extract_features(x)
        logits = self.classifier(feats)
        if self.is_binary_task:
            return logits[:, 0]
        return logits

    def compute_loss(self, pred, target, default_criterion):
        if self.is_binary_task:
            return F.binary_cross_entropy_with_logits(pred, target.float())
        return default_criterion(pred, target)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("[CodeBrain] Backbone frozen")

    def get_param_groups(self):
        encoder_lr_scale = float(getattr(self.params, "encoder_lr_scale", 1.0))
        classifier_lr_scale = float(getattr(self.params, "classifier_lr_scale", 10.0))
        return [
            {"params": list(self.backbone.parameters()), "lr_scale": encoder_lr_scale},
            {"params": list(self.classifier.parameters()), "lr_scale": classifier_lr_scale},
        ]
