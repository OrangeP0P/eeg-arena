"""
BENDR Model Wrapper - Adapts BENDR to the unified framework interface
"""
import os
import sys
import torch
import torch.nn as nn

# Import from our standalone bendr_core module (no dn3 dependencies)
# Use relative import since this is part of a package
try:
    from .bendr_core import ConvEncoderBENDR, BENDRContextualizer
except ImportError:
    # Fallback for when running directly
    from bendr_core import ConvEncoderBENDR, BENDRContextualizer


class Model(nn.Module):
    """
    BENDR Model Wrapper for unified framework.

    Input: Raw EEG data of shape (batch, channels, time_samples)
    Output: logits of shape (batch, num_classes)

    Unified contract:
    - forward(x) returns logits [B, num_classes]
    - extract_features(x) returns embeddings
    - freeze_backbone() freezes backbone parameters
    - get_param_groups() returns optimizer parameter groups
    """

    def __init__(self, params):
        super().__init__()
        self.params = params

        # dataset-provided metadata (auto-filled by data loader)
        num_classes = getattr(params, "num_of_classes", 2)
        num_channels = getattr(params, "num_channels", 64)
        samples = getattr(params, "samples", 1000)  # default if not provided

        # BENDR model hyperparameters
        encoder_h = getattr(params, "encoder_h", 512)
        contextualizer_hidden = getattr(params, "contextualizer_hidden", 3076)
        nhead = getattr(params, "nhead", 8)
        n_layer = getattr(params, "n_layer", 8)
        dropout = getattr(params, "dropout", 0.15)

        # Build BENDR backbone
        # ConvEncoderBENDR: (B, C, T) -> (B, encoder_h, downsampled_seq)
        self.encoder = ConvEncoderBENDR(
            in_features=num_channels,
            encoder_h=encoder_h,
            dropout=dropout,
            projection_head=False
        )

        # Calculate downsampled sequence length
        encoded_samples = self.encoder.downsampling_factor(samples)

        # BENDRContextualizer: (B, encoder_h, seq) -> (B, encoder_h, seq)
        self.contextualizer = BENDRContextualizer(
            in_features=encoder_h,
            hidden_feedforward=contextualizer_hidden,
            heads=nhead,
            layers=n_layer,
            dropout=dropout,
            finetuning=True  # Enable masking augmentation during finetuning (matches original BENDR)
        )

        # Pooling layer
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Classifier head
        self.classifier = nn.Linear(encoder_h, num_classes)

        # Load pretrained weights if enabled
        if getattr(params, "use_pretrained_weights", False):
            foundation_dir = getattr(params, "foundation_dir", "./models/Weights/BENDR")

            # BENDR has two pretrained files: encoder and contextualizer
            encoder_file = os.path.join(foundation_dir, "encoder.pt")
            contextualizer_file = os.path.join(foundation_dir, "contextualizer.pt")

            # Load contextualizer weights (always works - doesn't depend on channel count)
            if os.path.exists(contextualizer_file):
                try:
                    contextualizer_state = torch.load(contextualizer_file, map_location='cpu', weights_only=True)
                    self.contextualizer.load_state_dict(contextualizer_state, strict=False)
                    print(f"[BENDR] Loaded contextualizer weights from {contextualizer_file}")
                except Exception as e:
                    print(f"[BENDR] Warning: Failed to load contextualizer weights: {e}")
            else:
                print(f"[BENDR] Warning: Contextualizer weights not found at {contextualizer_file}")

            # Load encoder weights (may fail due to channel count mismatch)
            if os.path.exists(encoder_file):
                try:
                    encoder_state = torch.load(encoder_file, map_location='cpu', weights_only=True)

                    # Check if channel count matches
                    expected_in_channels = num_channels
                    # Try to detect the channel count from the checkpoint
                    ckpt_in_channels = None
                    for key in encoder_state.keys():
                        if 'encoder.Encoder_0.0.weight' in key:
                            ckpt_in_channels = encoder_state[key].shape[1]
                            break

                    if ckpt_in_channels is not None and ckpt_in_channels != expected_in_channels:
                        print(f"[BENDR] Warning: Pretrained encoder was trained with {ckpt_in_channels} channels, "
                              f"but dataset has {expected_in_channels} channels.")
                        print(f"[BENDR] Encoder will be initialized randomly (contextualizer pretrained weights loaded)")

                        # Try to load only matching parts (encoder layers after the first)
                        matched_state = {}
                        for key, value in encoder_state.items():
                            # Skip the first conv layer weights (channel mismatch)
                            if 'Encoder_0.0.weight' in key or 'Encoder_0.0.bias' in key:
                                continue
                            matched_state[key] = value

                        if matched_state:
                            # noinspection PyBroadException
                            try:
                                self.encoder.load_state_dict(matched_state, strict=False)
                                print(f"[BENDR] Loaded {len(matched_state)} matching encoder parameters (from layers 1-5)")
                            except Exception:
                                pass  # Ignore any remaining mismatches
                    else:
                        # Channel counts match, load normally
                        self.encoder.load_state_dict(encoder_state, strict=False)
                        print(f"[BENDR] Loaded encoder weights from {encoder_file}")
                except Exception as e:
                    print(f"[BENDR] Warning: Failed to load encoder weights: {e}")
            else:
                print(f"[BENDR] Warning: Encoder weights not found at {encoder_file}")

        # Freeze backbone if specified
        if getattr(params, "frozen", False):
            self.freeze_backbone()

    def extract_features(self, x):
        """
        Extract features from raw EEG input.

        Args:
            x: Raw EEG data of shape (batch, channels, time_samples)

        Returns:
            features of shape (batch, encoder_h)
        """
        # Encode: (B, C, T) -> (B, encoder_h, downsampled_seq)
        encoded = self.encoder(x)

        # Contextualize: (B, encoder_h, seq) -> (B, encoder_h, seq)
        context = self.contextualizer(encoded)

        # Pool: (B, encoder_h, seq) -> (B, encoder_h, 1)
        pooled = self.pool(context)

        # Squeeze: (B, encoder_h, 1) -> (B, encoder_h)
        features = pooled.squeeze(-1)

        return features

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x: Raw EEG data of shape (batch, channels, time_samples)

        Returns:
            logits of shape (batch, num_classes)
        """
        features = self.extract_features(x)
        logits = self.classifier(features)
        return logits

    def freeze_backbone(self):
        """Freeze backbone parameters when --frozen is enabled."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.contextualizer.parameters():
            param.requires_grad = False
        print("[BENDR] Backbone frozen (encoder + contextualizer)")

    def get_param_groups(self):
        """
        Return optimizer parameter groups.
        lr_scale is read from params (encoder_lr_scale, classifier_lr_scale).
        """
        encoder_lr_scale = getattr(self.params, 'encoder_lr_scale', 1.0)
        classifier_lr_scale = getattr(self.params, 'classifier_lr_scale', 5.0)
        return [
            {"params": list(self.encoder.parameters()) + list(self.contextualizer.parameters()), "lr_scale": encoder_lr_scale},
            {"params": self.classifier.parameters(), "lr_scale": classifier_lr_scale},
        ]
