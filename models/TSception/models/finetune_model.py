"""TSception fine-tune wrapper."""
from models._shared import MLPClassifierHead, RawBaselineModelBase, TSceptionBackbone


class Model(RawBaselineModelBase):
    def __init__(self, params):
        super().__init__(params)
        num_classes = getattr(params, "num_of_classes", 2)
        num_channels = getattr(params, "num_channels", 64)
        dropout = getattr(params, "dropout", 0.5)
        self.backbone = TSceptionBackbone(
            num_channels=num_channels,
            sampling_rate=getattr(params, "sampling_rate", 200),
            num_T=getattr(params, "num_T", 15),
            num_S=getattr(params, "num_S", 15),
        )
        self.classifier = MLPClassifierHead(
            self.backbone.feature_dim,
            getattr(params, "classifier_hidden", 32),
            num_classes,
            dropout=dropout,
        )

    def extract_features(self, x):
        return self.backbone(x)
