"""MSCFormer fine-tune wrapper."""
from models._shared import LinearClassifierHead, MSCFormerBackbone, RawBaselineModelBase


class Model(RawBaselineModelBase):
    def __init__(self, params):
        super().__init__(params)
        num_classes = getattr(params, "num_of_classes", 2)
        num_channels = getattr(params, "num_channels", 64)
        samples = getattr(params, "samples", 1000)
        self.backbone = MSCFormerBackbone(
            num_channels=num_channels,
            samples=samples,
            f1=getattr(params, "f1", 16),
            pooling_size=getattr(params, "pooling_size", 52),
            heads=getattr(params, "heads", 8),
            depth=getattr(params, "depth", 5),
            dropout=getattr(params, "dropout", 0.25),
        )
        self.classifier = LinearClassifierHead(self.backbone.feature_dim, num_classes)

    def extract_features(self, x):
        return self.backbone(x)
