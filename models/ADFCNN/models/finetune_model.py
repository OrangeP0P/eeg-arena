"""ADFCNN fine-tune wrapper."""
from models._shared import ADFCNNBackbone, LinearWithConstraint, RawBaselineModelBase


class Model(RawBaselineModelBase):
    def __init__(self, params):
        super().__init__(params)
        num_classes = getattr(params, "num_of_classes", 2)
        num_channels = getattr(params, "num_channels", 64)
        self.backbone = ADFCNNBackbone(
            num_channels=num_channels,
            dropout=getattr(params, "dropout", 0.25),
            f1=getattr(params, "F1", 8),
            d=getattr(params, "D", 1),
        )
        self.classifier = LinearWithConstraint(self.backbone.feature_dim, num_classes, max_norm=0.5)

    def extract_features(self, x):
        return self.backbone(x)
