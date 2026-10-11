"""
REVE Model Definition for Fine-tuning

REVE (Recurrent Vision EEG) uses 4D positional embeddings (x, y, z, t)
and requires electrode position coordinates.
"""
import os
import torch
import torch.nn as nn
import numpy as np

from .modeling_reve import Reve as ReveBase
from ..config import REVE_BASE_CONFIG, REVE_LARGE_CONFIG


class Model(nn.Module):
    """
    REVE Model for Fine-tuning

    Input: [Batch, Channel, Time] - Raw EEG + position tensor
    Output: [Batch, num_classes]

    The model requires electrode position information which is passed
    via the pos_tensor attribute (set during initialization from adapter).
    """

    def __init__(self, params):
        super(Model, self).__init__()

        # Dataset metadata
        num_classes = getattr(params, 'num_of_classes', 2)
        num_channels = getattr(params, 'num_channels', 64)

        # Get model size (base or large)
        model_size = getattr(params, 'model_size', 'base')

        # Use preset config based on model_size
        if model_size == 'large':
            preset_config = REVE_LARGE_CONFIG
        else:
            preset_config = REVE_BASE_CONFIG

        # Model hyperparameters - use preset values as defaults
        embed_dim = getattr(params, 'embed_dim', preset_config['embed_dim'])
        depth = getattr(params, 'depth', preset_config['depth'])
        heads = getattr(params, 'heads', preset_config['heads'])
        head_dim = getattr(params, 'head_dim', preset_config['head_dim'])
        mlp_dim_ratio = getattr(params, 'mlp_dim_ratio', preset_config['mlp_dim_ratio'])
        use_geglu = getattr(params, 'use_geglu', preset_config['use_geglu'])
        freqs = getattr(params, 'freqs', preset_config['freqs'])
        noise_ratio = getattr(params, 'noise_ratio', preset_config['noise_ratio'])
        patch_size = getattr(params, 'patch_size', preset_config['patch_size'])
        patch_overlap = getattr(params, 'patch_overlap', preset_config['patch_overlap'])
        dropout = getattr(params, 'dropout', 0.1)

        self.params = params
        self.num_channels = num_channels
        self.model_size = model_size

        # Create REVE config
        from .configuration_reve import ReveConfig
        config = ReveConfig(
            embed_dim=embed_dim,
            depth=depth,
            heads=heads,
            head_dim=head_dim,
            mlp_dim_ratio=mlp_dim_ratio,
            use_geglu=use_geglu,
            freqs=freqs,
            noise_ratio=noise_ratio,
            patch_size=patch_size,
            patch_overlap=patch_overlap,
        )

        # Build backbone
        self.backbone = ReveBase(config)

        # Load pretrained weights
        if getattr(params, 'use_pretrained_weights', True):
            foundation_dir = getattr(
                params,
                'foundation_dir',
                './models/Weights/REVE'
            )

            # Resolve path relative to project root if needed
            if not os.path.isabs(foundation_dir):
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
                foundation_dir = os.path.join(project_root, foundation_dir)

            # Determine pretrained weights file
            if os.path.isfile(foundation_dir):
                # foundation_dir is already a full file path (e.g. resolved by --variant)
                pretrained_file = foundation_dir
            else:
                # foundation_dir is a directory, construct filename from model_size
                pretrained_file = os.path.join(foundation_dir, f'reve-{model_size}.safetensors')

                if not os.path.exists(pretrained_file):
                    # Fallback: try .pth format
                    pth_file = os.path.join(foundation_dir, f'reve-{model_size}.pth')
                    if os.path.exists(pth_file):
                        pretrained_file = pth_file
                    else:
                        print(f'[REVE] Warning: Pretrained weights not found at {pretrained_file}')
                        pretrained_file = None

            if pretrained_file:
                try:
                    # Load safetensors or pth
                    if pretrained_file.endswith('.safetensors'):
                        from safetensors.torch import load_file
                        state_dict = load_file(pretrained_file, device='cpu')
                    else:
                        state_dict = torch.load(pretrained_file, map_location='cpu', weights_only=True)

                    # Handle different checkpoint formats
                    if 'model' in state_dict:
                        state_dict = state_dict['model']
                    elif 'state_dict' in state_dict:
                        state_dict = state_dict['state_dict']
                    elif 'model_state_dict' in state_dict:
                        state_dict = state_dict['model_state_dict']

                    # Keep only backbone parameters. Pretraining checkpoints may
                    # store them under ``encoder.`` alongside decoder weights,
                    # while downstream checkpoints use the backbone keys directly.
                    backbone_keys = set(self.backbone.state_dict().keys())
                    backbone_state_dict = {}
                    for key, value in state_dict.items():
                        key = key.replace('module.', '')
                        if key.startswith('encoder.'):
                            key = key[len('encoder.'):]
                        if key in backbone_keys:
                            backbone_state_dict[key] = value

                    if not backbone_state_dict:
                        raise ValueError('Checkpoint contains no REVE backbone parameters')

                    msg = self.backbone.load_state_dict(backbone_state_dict, strict=False)
                    print(
                        f'[REVE] Loaded {len(backbone_state_dict)}/{len(backbone_keys)} '
                        f'backbone tensors from {pretrained_file}'
                    )
                    if msg.missing_keys:
                        print(f'[REVE] Missing keys: {msg.missing_keys[:5]}...')
                    if msg.unexpected_keys:
                        print(f'[REVE] Unexpected keys: {msg.unexpected_keys[:5]}...')
                except Exception as e:
                    print(f'[REVE] Warning: Failed to load pretrained weights: {e}')
            else:
                print(f'[REVE] Warning: Pretrained weights not found in {foundation_dir}')

        # Freeze backbone if specified
        if getattr(params, 'frozen', False):
            self.freeze_backbone()

        # Build classifier head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

        self.embed_dim = embed_dim

        # Store ch_positions for later use in forward
        self._ch_positions = getattr(params, 'ch_positions', None)

    def _get_pos_tensor(self, device):
        """
        Convert ch_positions dictionary to position tensor

        Returns:
            pos_tensor: (num_channels, 3)
        """
        ch_positions = self._ch_positions

        if ch_positions is None:
            # Fallback: create default positions if not provided
            print('[REVE] Warning: ch_positions not provided, using default zeros')
            return torch.zeros(self.num_channels, 3, device=device)

        # Convert dictionary to array
        positions = []
        for ch_name in ch_positions.keys():
            pos = ch_positions[ch_name]
            if pos is None:
                pos = [0.0, 0.0, 0.0]
            positions.append(pos)

        # Ensure we have the right number of positions
        if len(positions) < self.num_channels:
            while len(positions) < self.num_channels:
                positions.append([0.0, 0.0, 0.0])
        elif len(positions) > self.num_channels:
            positions = positions[:self.num_channels]

        pos_tensor = torch.from_numpy(np.array(positions, dtype=np.float32))
        return pos_tensor.to(device)

    def extract_features(self, x):
        """
        Extract backbone features

        Args:
            x: Input [Batch, Channel, Time] - Raw EEG

        Returns:
            Features [Batch, embed_dim]
        """
        # Get position tensor from params.ch_positions (set by FinetuneDataLoader)
        # Shape: (num_channels, 3) -> expand to (batch_size, num_channels, 3)
        pos = self._get_pos_tensor(x.device)  # (C, 3)
        pos = pos.unsqueeze(0).expand(x.size(0), -1, -1)  # (B, C, 3)
        # Convert to same dtype as input (for FlashAttention fp16 compatibility)
        pos = pos.to(dtype=x.dtype)

        # Forward through backbone
        # REVE forward returns (B, C, H, E) where H is number of patches
        out = self.backbone(x, pos, return_output=False)

        # Apply attention pooling to get (B, E)
        features = self.backbone.attention_pooling(out)

        return features

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input [Batch, Channel, Time] - Raw EEG

        Returns:
            Output [Batch, num_classes]
        """
        features = self.extract_features(x)
        logits = self.classifier(features)
        return logits

    def freeze_backbone(self):
        """
        Freeze backbone parameters for linear probing style fine-tuning.
        Only unfreeze the classifier head.
        """
        for name, param in self.backbone.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
        print('[REVE] Backbone frozen (except classification head)')

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

        # Add classifier head params
        for param in self.classifier.parameters():
            classifier_params.append(param)

        encoder_lr_scale = getattr(self.params, 'encoder_lr_scale', 1.0)
        classifier_lr_scale = getattr(self.params, 'classifier_lr_scale', 5.0)
        return [
            {"params": backbone_params, "lr_scale": encoder_lr_scale},
            {"params": classifier_params, "lr_scale": classifier_lr_scale},
        ]
