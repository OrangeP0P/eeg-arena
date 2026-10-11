"""
NeuroGPT Fine-tuning Model

This module wraps the NeuroGPT encoder for fine-tuning with a unified interface.
This is a simplified version that uses the EEGConformer encoder + a classification head.
"""
import os
import torch
import torch.nn as nn
from .upstream import Encoder


class Model(nn.Module):
    """
    NeuroGPT Fine-tuning Model (Simplified)

    Input: [Batch, Channel, Patch_Num, Patch_Size] = [B, C, S, D]
    Output: [Batch, num_classes]

    This simplified version uses:
    - Encoder: EEGConformer from braindecode (feature extraction mode)
    - Classifier: MLP classifier on pooled features
    """

    def __init__(self, params):
        super(Model, self).__init__()

        # Store params for later use (e.g., get_param_groups)
        self.params = params

        # Number of channels (EEG channels)
        self.num_channels = getattr(params, 'num_channels', 49)
        # Patch size
        self.patch_size = 200
        # Number of patches
        self.patch_num = getattr(params, 'patch_num', 30)

        # Encoder parameters
        encoder_n_filters_time = getattr(params, 'encoder_n_filters_time', 40)
        encoder_filter_time_length = getattr(params, 'encoder_filter_time_length', 25)
        encoder_pool_time_length = getattr(params, 'encoder_pool_time_length', 75)
        encoder_pool_time_stride = getattr(params, 'encoder_pool_time_stride', 15)
        encoder_drop_prob = getattr(params, 'encoder_drop_prob', 0.5)
        encoder_att_depth = getattr(params, 'encoder_att_depth', 6)
        encoder_att_heads = getattr(params, 'encoder_att_heads', 10)
        encoder_att_drop_prob = getattr(params, 'encoder_att_drop_prob', 0.5)

        # Number of classes
        num_classes = getattr(params, 'num_of_classes', 2)

        # Build encoder in non-decoding mode (for feature extraction)
        self.encoder = Encoder(
            n_outputs=4,  # Dummy value, not used
            n_chans=self.num_channels,
            n_times=self.patch_size,
            n_filters_time=encoder_n_filters_time,
            filter_time_length=encoder_filter_time_length,
            pool_time_length=encoder_pool_time_length,
            pool_time_stride=encoder_pool_time_stride,
            drop_prob=encoder_drop_prob,
            att_depth=encoder_att_depth,
            att_heads=encoder_att_heads,
            att_drop_prob=encoder_att_drop_prob,
            is_decoding_mode=False,  # Feature extraction mode
        )

        # Calculate encoder output size
        # The encoder outputs [B*S, seq_len, embed_dim] after the transformer
        # We need to calculate the seq_len based on the encoder parameters
        # For now, use a known size or make it dynamic
        self.encoder_output_dim = encoder_n_filters_time  # Base embedding dim

        # For classification, we use a simple MLP
        # Input: [B, S, encoder_output_dim] (after pooling across patches)
        self.classifier = nn.Sequential(
            nn.Linear(encoder_n_filters_time, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

        self.num_classes = num_classes

        # Load pretrained weights
        if getattr(params, 'use_pretrained_weights', False):
            foundation_dir = getattr(params, 'foundation_dir', './models/Weights/NeuroGPT/pytorch_model.bin')
            map_location = getattr(params, 'device_obj', torch.device('cpu'))

            # Convert relative path to absolute path if needed
            if not os.path.isabs(foundation_dir):
                # Get project root (parent of models directory)
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
                foundation_dir = os.path.join(project_root, foundation_dir)

            if os.path.exists(foundation_dir):
                try:
                    checkpoint = torch.load(foundation_dir, map_location=map_location, weights_only=True)
                    # Handle different checkpoint formats
                    if 'model' in checkpoint:
                        self.encoder.load_state_dict(checkpoint['model'], strict=False)
                    elif 'state_dict' in checkpoint:
                        self.encoder.load_state_dict(checkpoint['state_dict'], strict=False)
                    else:
                        self.encoder.load_state_dict(checkpoint, strict=False)
                    print(f"[NeuroGPT] Loaded pretrained weights from {foundation_dir}")
                except Exception as e:
                    print(f"[NeuroGPT] Warning: Failed to load pretrained weights from {foundation_dir}: {e}")
            else:
                print(f"[NeuroGPT] Warning: Pretrained weights not found at {foundation_dir}")

        if getattr(params, 'frozen', False):
            self.freeze_backbone()

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input [Batch, Channel, Patch_Num, Patch_Size] = [B, C, S, D]

        Returns:
            Output [Batch, num_classes]
        """
        batch_size, num_channels, num_patches, patch_size = x.size()

        # The encoder expects [B, S, C, D] but we have [B, C, S, D]
        # Rearrange dimensions: [B, C, S, D] -> [B, S, C, D]
        x = x.permute(0, 2, 1, 3)

        # Use the encoder's forward pass (feature extraction mode)
        # Output shape: [B * S, seq_len, embed_dim]
        features = self.encoder(x)

        # Reshape: [B * S, seq_len, embed_dim] -> [B, S, seq_len, embed_dim]
        features = features.view(batch_size, num_patches, features.size(1), features.size(2))

        # Average pooling across patches and sequence
        # [B, S, seq_len, embed_dim] -> [B, embed_dim]
        features = features.mean(dim=(1, 2))

        # Pass through classifier
        output = self.classifier(features)

        return output

    def extract_features(self, x):
        """Return backbone features for evaluator or downstream heads."""
        batch_size, num_channels, num_patches, patch_size = x.size()

        # The encoder expects [B, S, C, D] but we have [B, C, S, D]
        x = x.permute(0, 2, 1, 3)

        # Use the encoder's forward pass (feature extraction mode)
        features = self.encoder(x)

        # Reshape: [B * S, seq_len, embed_dim] -> [B, S, seq_len, embed_dim]
        features = features.view(batch_size, num_patches, features.size(1), features.size(2))

        # Average pooling across patches and sequence
        features = features.mean(dim=(1, 2))

        return features

    def freeze_backbone(self):
        """Freeze backbone parameters for linear probing style fine-tuning."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def get_param_groups(self):
        """
        Return optimizer parameter groups.
        lr_scale is read from params (encoder_lr_scale, classifier_lr_scale).
        """
        encoder_lr_scale = getattr(self.params, 'encoder_lr_scale', 1.0)
        classifier_lr_scale = getattr(self.params, 'classifier_lr_scale', 5.0)
        return [
            {"params": list(self.encoder.parameters()), "lr_scale": encoder_lr_scale},
            {"params": list(self.classifier.parameters()), "lr_scale": classifier_lr_scale},
        ]
