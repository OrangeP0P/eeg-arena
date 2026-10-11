"""YASA-style fine-tune wrapper."""
from models._shared import RawBaselineModelBase, YASABackbone
from models._shared.classifier_heads import LinearClassifierHead


class Model(RawBaselineModelBase):
    def __init__(self, params):
        super().__init__(params)
        num_classes = getattr(params, "num_of_classes", 2)
        self.backbone = YASABackbone(
            hidden_dim=getattr(params, "hidden_dim", 128),
            sample_rate=getattr(params, "sampling_rate", 200),
            dropout=getattr(params, "dropout", 0.2),
        )
        self.classifier = LinearClassifierHead(self.backbone.feature_dim, num_classes)

    def extract_features(self, x):
        return self.backbone(x)
