"""
CBraMod Model Definition
"""
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from .cbramod import CBraMod


class Model(nn.Module):
    """
    CBraMod Model

    Input: [Batch, Channel, Patch_Num, Patch_Size]
    Output: [Batch, num_classes]
    """

    def __init__(self, params):
        super(Model, self).__init__()

        # Number of channels
        self.channel_num = getattr(params, 'num_channels', 49)

        # Patch configuration
        self.patch_num = getattr(params, 'patch_num', 30)
        self.patch_size = 200
        self.flatten_dim = self.channel_num * self.patch_num * self.patch_size

        # Model parameters - get from config or use default values
        d_model = getattr(params, 'd_model', 200)
        nhead = getattr(params, 'nhead', 8)
        n_layer = getattr(params, 'n_layer', 12)
        dim_feedforward = getattr(params, 'dim_feedforward', 800)

        # Backbone
        self.backbone = CBraMod(
            in_dim=d_model,
            out_dim=d_model,
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            patch_num=self.patch_num,
            n_layer=n_layer,
            nhead=nhead
        )

        # Load pretrained weights
        if getattr(params, 'use_pretrained_weights', False):
            foundation_dir = getattr(params, 'foundation_dir', './models/Weights/CBraMod/pretrained_weights.pth')
            map_location = getattr(params, "device_obj", torch.device("cpu"))
            self.backbone.load_state_dict(torch.load(foundation_dir, map_location=map_location, weights_only=True))

        # Remove the original projection layer
        self.backbone.proj_out = nn.Identity()

        # Classifier
        self.classifier = self._build_classifier(params)

    def _build_classifier(self, params):
        """Build classifier"""
        classifier_type = getattr(params, 'classifier', 'all_patch_reps')
        dropout = getattr(params, 'dropout', 0.1)
        num_classes = getattr(params, 'num_of_classes', 2)
        d_model = getattr(params, 'd_model', 200)

        if classifier_type == 'avgpooling_patch_reps':
            return nn.Sequential(
                Rearrange('b c s d -> b d c s'),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(d_model, num_classes),
            )
        elif classifier_type == 'all_patch_reps_onelayer':
            return nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(self.flatten_dim, num_classes),
            )
        elif classifier_type == 'all_patch_reps_twolayer':
            return nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(self.flatten_dim, d_model),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, num_classes),
            )
        elif classifier_type == 'all_patch_reps':
            return nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(self.flatten_dim, 4 * d_model),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(4 * d_model, d_model),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, num_classes),
            )
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input [Batch, Channel, Patch_Num, Patch_Size]

        Returns:
            Output [Batch, num_classes]
        """
        feats = self.extract_features(x)
        logits = self.classifier(feats)
        return logits

    def extract_features(self, x):
        """Return backbone features for evaluator or downstream heads."""
        return self.backbone(x)

    def freeze_backbone(self):
        """Freeze backbone parameters for linear probing style fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def get_param_groups(self):
        """
        Return optimizer parameter groups.
        lr_scale keeps previous behavior: classifier uses 5x learning rate.
        """
        return [
            {"params": list(self.backbone.parameters()), "lr_scale": 1.0},
            {"params": list(self.classifier.parameters()), "lr_scale": 5.0},
        ]
