"""USleep fine-tune wrapper."""
from models._shared import MLPClassifierHead, RawBaselineModelBase, USleepBackbone


class Model(RawBaselineModelBase):
    def __init__(self, params):
        super().__init__(params)
        num_classes = getattr(params, "num_of_classes", 2)
        num_channels = getattr(params, "num_channels", 1)
        dropout = getattr(params, "dropout", 0.25)
        self.backbone = USleepBackbone(
            num_channels=num_channels,
            base_filters=getattr(params, "base_filters", 16),
            depth=getattr(params, "depth", 4),
        )
        self.classifier = MLPClassifierHead(
            self.backbone.feature_dim,
            getattr(params, "classifier_hidden", 64),
            num_classes,
            dropout=dropout,
        )

    def extract_features(self, x):
        return self.backbone(x)
