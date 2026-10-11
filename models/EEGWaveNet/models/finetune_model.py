"""EEGWaveNet fine-tune wrapper."""
import torch.nn as nn

from models._shared import EEGWaveNetBackbone, RawBaselineModelBase


class Model(RawBaselineModelBase):
    def __init__(self, params):
        super().__init__(params)
        num_classes = getattr(params, "num_of_classes", 2)
        num_channels = getattr(params, "num_channels", 64)
        self.backbone = EEGWaveNetBackbone(num_channels=num_channels)
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.feature_dim, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, 32),
            nn.Sigmoid(),
            nn.Linear(32, num_classes),
        )

    def extract_features(self, x):
        return self.backbone(x)
