"""
BIOT Model Definition for Fine-tuning

BIOT (Biosignal Transformer) converts EEG signals to "sentence" structure
and uses Transformer encoding in the frequency domain via STFT.
"""
import os
import torch
import torch.nn as nn
from .biot import BIOTClassifier


class Model(nn.Module):
    """
    BIOT Model for Fine-tuning

    Input: [Batch, Channel, Time] - Raw EEG
    Output: [Batch, num_classes]
    """

    def __init__(self, params):
        super(Model, self).__init__()

        # Store params for later use (e.g., get_param_groups)
        self.params = params

        # Number of classes
        num_classes = getattr(params, 'num_of_classes', 2)

        # Number of channels
        num_channels = getattr(params, 'num_channels', 16)

        # Model parameters
        emb_size = getattr(params, 'emb_size', 256)
        heads = getattr(params, 'heads', 8)
        depth = getattr(params, 'depth', 4)
        dropout = getattr(params, 'dropout', 0.1)
        n_fft = getattr(params, 'n_fft', 200)
        hop_length = getattr(params, 'hop_length', 100)

        # Backbone - BIOT Classifier
        self.backbone = BIOTClassifier(
            emb_size=emb_size,
            heads=heads,
            depth=depth,
            n_classes=num_classes,
            n_channels=num_channels,
            n_fft=n_fft,
            hop_length=hop_length,
        )

        # Load pretrained weights
        if getattr(params, 'use_pretrained_weights', False):
            foundation_dir = getattr(
                params,
                'foundation_dir',
                './models/Weights/BIOT/EEG-SHHS+PREST-18-channels.ckpt'
            )

            # Resolve path relative to project root if needed
            if not os.path.isabs(foundation_dir):
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
                foundation_dir = os.path.join(project_root, foundation_dir)

            if os.path.exists(foundation_dir):
                map_location = getattr(params, "device_obj", torch.device("cpu"))
                state_dict = torch.load(foundation_dir, map_location=map_location, weights_only=True)

                # Handle different checkpoint formats
                if 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']
                elif 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']

                # Load state dict, ignoring mismatched keys
                try:
                    self.backbone.load_state_dict(state_dict, strict=False)
                    print(f"Loaded pretrained weights from {foundation_dir}")
                except Exception as e:
                    print(f"Warning: Failed to load pretrained weights from {foundation_dir}: {e}")
            else:
                print(f"Warning: Pretrained weights not found at {foundation_dir}")

        # Store embedding size for feature extraction
        self.emb_size = emb_size

        if getattr(params, 'frozen', False):
            self.freeze_backbone()

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input [Batch, Channel, Time] - Raw EEG

        Returns:
            Output [Batch, num_classes]
        """
        return self.backbone(x)

    def extract_features(self, x):
        """
        Extract backbone features (before classifier head)

        Args:
            x: Input [Batch, Channel, Time] - Raw EEG

        Returns:
            Features [Batch, emb_size]
        """
        return self.backbone.biot(x)

    def freeze_backbone(self):
        """
        Freeze backbone parameters for linear probing style fine-tuning.
        Only unfreeze the classifier head.
        """
        for name, param in self.backbone.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False

    def get_param_groups(self):
        """
        Return optimizer parameter groups.
        lr_scale is read from params (encoder_lr_scale, classifier_lr_scale).
        """
        backbone_params = []
        classifier_params = []

        for name, param in self.backbone.named_parameters():
            if 'classifier' in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)

        encoder_lr_scale = getattr(self.params, 'encoder_lr_scale', 1.0)
        classifier_lr_scale = getattr(self.params, 'classifier_lr_scale', 5.0)
        return [
            {"params": backbone_params, "lr_scale": encoder_lr_scale},
            {"params": classifier_params, "lr_scale": classifier_lr_scale},
        ]
