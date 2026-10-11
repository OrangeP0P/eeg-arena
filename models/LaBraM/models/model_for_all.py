"""
LaBraM Model Wrapper - Adapts LaBraM to the unified framework interface
"""
import os
import sys
import torch
import torch.nn as nn

# Add current directory to path for importing modeling_finetune
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from modeling_finetune import NeuralTransformer


class Model(nn.Module):
    """
    Unified model contract:
    - forward(x) returns logits [B, num_classes]
    - extract_features(x) is optional but recommended

    LaBraM expects patched EEG input of shape (batch, channels, patch_num, patch_size)
    For base model: patch_size = 200

    Note: LaBraM has specific positional embedding requirements.
    This wrapper uses PatchEmbed directly and skips problematic position embeddings.
    """

    def __init__(self, params):
        super().__init__()
        self.params = params

        # dataset-provided metadata (auto-filled by data loader)
        num_classes = getattr(params, "num_of_classes", 2)
        num_channels = getattr(params, "num_channels", 64)

        # LaBraM model hyperparameters
        dropout = getattr(params, "dropout", 0.0)

        # Build LaBraM backbone without positional embeddings
        # Using in_chans=num_channels to use PatchEmbed which handles channel dimension properly
        # We set EEG_size to match the expected input: num_channels * patch_num * patch_size
        # For standard 8 patches with 64 channels: 64 * 8 * 200 = 102400, round to multiple of patch_size
        patch_size = 200

        # Create a simplified version that avoids positional embedding issues
        # We'll add our own classifier head
        self.backbone = NeuralTransformer(
            EEG_size=num_channels * 8 * patch_size,  # Large EEG size to accommodate various channel counts
            patch_size=patch_size,
            in_chans=num_channels,
            num_classes=num_classes,
            drop_rate=dropout,
            attn_drop_rate=dropout,
            init_values=0,
            use_abs_pos_emb=False,  # Disable positional embeddings
        )

        # Replace time_embed with None to avoid dimension issues
        self.backbone.time_embed = None

        # Load pretrained weights if enabled
        if getattr(params, "use_pretrained_weights", False):
            foundation_dir = getattr(params, "foundation_dir", "./models/Weights/LaBraM")
            pretrained_file = os.path.join(foundation_dir, "labram-base.pth")

            if os.path.exists(pretrained_file):
                checkpoint = torch.load(pretrained_file, map_location='cpu')
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint

                msg = self.backbone.load_state_dict(state_dict, strict=False)
                print(f"[LaBraM] Loaded pretrained weights from {pretrained_file}")
                print(f"[LaBraM] Missing keys: {msg.missing_keys}")
                print(f"[LaBraM] Unexpected keys: {msg.unexpected_keys}")
            else:
                print(f"[LaBraM] Warning: Pretrained weights not found at {pretrained_file}")

        # Freeze backbone if specified
        if getattr(params, "frozen", False):
            self.freeze_backbone()

    def extract_features(self, x):
        """
        Extract features from patched EEG input.

        Args:
            x: patched EEG data of shape (batch, channels, patch_num, patch_size)

        Returns:
            features of shape (batch, embed_dim)
        """
        # Use forward_features with disabled position embeddings
        features = self.backbone.forward_features(x)
        return features

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x: patched EEG data of shape (batch, channels, patch_num, patch_size)

        Returns:
            logits of shape (batch, num_classes)
        """
        # LaBraM forward already includes classification head
        logits = self.backbone(x)
        return logits

    def freeze_backbone(self):
        """Freeze backbone parameters when --frozen is enabled."""
        for name, param in self.backbone.named_parameters():
            if 'head' not in name:
                param.requires_grad = False
        print("[LaBraM] Backbone frozen (except classification head)")

    def get_param_groups(self):
        """
        Optional API for optimizer parameter grouping.
        Trainer will prioritize this over name-based matching.
        """
        backbone_params = []
        head_params = []

        for name, param in self.backbone.named_parameters():
            if 'head' in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

        return [
            {"params": backbone_params, "lr_scale": 1.0},
            {"params": head_params, "lr_scale": 10.0},
        ]
