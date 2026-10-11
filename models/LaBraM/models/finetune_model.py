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

    Note: LaBraM uses positional embeddings indexed by channel position in its pretrained space.
    The data loader provides model_channel_indices for correct pos_embed lookup.
    """

    def __init__(self, params):
        super().__init__()
        self.params = params

        # dataset-provided metadata (auto-filled by data loader)
        num_classes = getattr(params, "num_of_classes", 2)
        num_channels = getattr(params, "num_channels", 64)

        # LaBraM model hyperparameters
        dropout = getattr(params, "dropout", 0.0)

        # Build LaBraM backbone matching pretrained architecture
        # LaBraM was pretrained with in_chans=1 → TemporalConv path
        # TemporalConv processes each (channel, time_window) independently, producing
        # num_channels * time_windows tokens, enabling per-channel pos_embed indexing.
        # Using in_chans=num_channels would select PatchEmbed (Conv2d) which collapses
        # all channels into embed_dim, breaking pos_embed's per-channel design.
        patch_size = 200

        self.backbone = NeuralTransformer(
            EEG_size=num_channels * 8 * patch_size,
            patch_size=patch_size,
            in_chans=1,  # Use TemporalConv (matches pretrained architecture)
            num_classes=num_classes,
            drop_rate=dropout,
            attn_drop_rate=dropout,
            init_values=0,
            use_abs_pos_emb=True,  # Enable positional embeddings with channel indices
        )

        # Compute input_chans for pos_embed indexing
        # pos_embed shape: [1, 129, embed_dim] where index 0 = CLS, indices 1-128 = channel positions
        model_ch_indices = getattr(params, 'model_channel_indices', None)
        if model_ch_indices is not None:
            # input_chans selects from pos_embed: [CLS_idx=0, ch1_idx+1, ch2_idx+1, ...]
            self.input_chans = [0] + [idx + 1 for idx in model_ch_indices]
        else:
            self.input_chans = None

        # Load pretrained weights if enabled
        if getattr(params, "use_pretrained_weights", False):
            foundation_dir = getattr(params, "foundation_dir", "./models/Weights/LaBraM")

            # Support both file path and directory path
            if os.path.isfile(foundation_dir):
                pretrained_file = foundation_dir
            else:
                pretrained_file = os.path.join(foundation_dir, "labram-base.pth")

            if os.path.exists(pretrained_file):
                checkpoint = torch.load(pretrained_file, map_location='cpu', weights_only=False)

                # Extract state_dict based on checkpoint format
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint

                # Handle checkpoint with 'student.' prefix (from distillation training)
                new_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith('student.'):
                        new_key = key[8:]  # Remove 'student.' prefix
                    else:
                        new_key = key
                    new_state_dict[new_key] = value

                msg = self.backbone.load_state_dict(new_state_dict, strict=False)
                print(f"[LaBraM] Loaded pretrained weights from {pretrained_file}")
                if msg.missing_keys:
                    print(f"[LaBraM] Missing keys: {msg.missing_keys[:5]}... (total {len(msg.missing_keys)})")
                if msg.unexpected_keys:
                    print(f"[LaBraM] Unexpected keys: {msg.unexpected_keys[:5]}... (total {len(msg.unexpected_keys)})")
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
        features = self.backbone.forward_features(x, input_chans=self.input_chans)
        return features

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x: patched EEG data of shape (batch, channels, patch_num, patch_size)

        Returns:
            logits of shape (batch, num_classes)
        """
        logits = self.backbone(x, input_chans=self.input_chans)
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
        lr_scale is read from params (encoder_lr_scale, classifier_lr_scale).
        """
        backbone_params = []
        head_params = []

        for name, param in self.backbone.named_parameters():
            if 'head' in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

        encoder_lr_scale = getattr(self.params, 'encoder_lr_scale', 1.0)
        classifier_lr_scale = getattr(self.params, 'classifier_lr_scale', 10.0)
        return [
            {"params": backbone_params, "lr_scale": encoder_lr_scale},
            {"params": head_params, "lr_scale": classifier_lr_scale},
        ]
