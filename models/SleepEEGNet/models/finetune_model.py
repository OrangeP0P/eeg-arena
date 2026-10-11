"""SleepEEGNet fine-tune wrapper."""
from models._shared import MLPClassifierHead, RawBaselineModelBase, SleepEEGNetBackbone


class Model(RawBaselineModelBase):
    def __init__(self, params):
        super().__init__(params)
        hidden_size = getattr(params, "hidden_size", 128)
        dropout = getattr(params, "dropout", 0.5)
        num_classes = getattr(params, "num_of_classes", 2)
        num_channels = getattr(params, "num_channels", 1)
        self.backbone = SleepEEGNetBackbone(
            num_channels=num_channels,
            target_steps=getattr(params, "target_steps", 10),
            hidden_size=hidden_size,
            dropout=dropout,
        )
        self.classifier = MLPClassifierHead(self.backbone.feature_dim, hidden_size, num_classes, dropout=dropout)

    def extract_features(self, x):
        return self.backbone(x)
