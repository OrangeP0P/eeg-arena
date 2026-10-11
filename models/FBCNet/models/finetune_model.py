"""FBCNet fine-tune wrapper."""
from models._shared import FBCNetBackbone, LinearWithConstraint, RawBaselineModelBase


class Model(RawBaselineModelBase):
    def __init__(self, params):
        super().__init__(params)
        num_classes = getattr(params, "num_of_classes", 2)
        num_channels = getattr(params, "num_channels", 64)
        self.backbone = FBCNetBackbone(
            num_channels=num_channels,
            m=getattr(params, "m", 16),
            stat_segments=getattr(params, "stat_segments", 4),
        )
        self.classifier = LinearWithConstraint(self.backbone.feature_dim, num_classes, max_norm=0.5)

    def extract_features(self, x):
        return self.backbone(x)
