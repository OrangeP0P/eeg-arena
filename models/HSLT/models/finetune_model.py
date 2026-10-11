"""HSLT fine-tune wrapper."""
from models._shared import HSLTBackbone, LinearClassifierHead, RawBaselineModelBase


class Model(RawBaselineModelBase):
    def __init__(self, params):
        super().__init__(params)
        num_classes = getattr(params, "num_of_classes", 2)
        num_channels = getattr(params, "num_channels", 64)
        self.backbone = HSLTBackbone(
            num_channels=num_channels,
            ch_positions=getattr(params, "ch_positions", None),
            feature_bins=getattr(params, "feature_bins", 5),
            De=getattr(params, "De", 8),
            Dr=getattr(params, "Dr", 16),
            Le=getattr(params, "Le", 2),
            Lr=getattr(params, "Lr", 2),
        )
        self.classifier = LinearClassifierHead(self.backbone.feature_dim, num_classes)

    def extract_features(self, x):
        return self.backbone(x)
