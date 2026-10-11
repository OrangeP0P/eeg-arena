"""SlimSeiz fine-tune wrapper."""
from models._shared import MLPClassifierHead, RawBaselineModelBase, SlimSeizBackbone


class Model(RawBaselineModelBase):
    def __init__(self, params):
        super().__init__(params)
        hidden_dim = getattr(params, "hidden_dim", 96)
        dropout = getattr(params, "dropout", 0.2)
        num_classes = getattr(params, "num_of_classes", 2)
        num_channels = getattr(params, "num_channels", 64)
        self.backbone = SlimSeizBackbone(
            num_channels=num_channels,
            hidden_dim=hidden_dim,
            depth=getattr(params, "depth", 4),
            dropout=dropout,
        )
        self.classifier = MLPClassifierHead(self.backbone.feature_dim, hidden_dim, num_classes, dropout=dropout)

    def extract_features(self, x):
        return self.backbone(x)
