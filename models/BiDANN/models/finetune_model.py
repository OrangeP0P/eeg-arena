"""BiDANN fine-tune wrapper."""
from models._shared import BiDANNBackbone, MLPClassifierHead, RawBaselineModelBase


class Model(RawBaselineModelBase):
    def __init__(self, params):
        super().__init__(params)
        num_classes = getattr(params, "num_of_classes", 2)
        num_channels = getattr(params, "num_channels", 64)
        hidden_size = getattr(params, "hidden_size", 128)
        self.backbone = BiDANNBackbone(
            num_channels=num_channels,
            ch_positions=getattr(params, "ch_positions", None),
            segments=getattr(params, "segments", 9),
            hidden_size=hidden_size,
            projected_steps=getattr(params, "projected_steps", 8),
            dropout=getattr(params, "dropout", 0.3),
        )
        self.classifier = MLPClassifierHead(self.backbone.feature_dim, hidden_size * 2, num_classes, dropout=0.3)

    def extract_features(self, x):
        return self.backbone(x)
