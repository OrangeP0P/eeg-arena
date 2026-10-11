"""TMSANet fine-tune wrapper."""
from models._shared import LinearClassifierHead, RawBaselineModelBase, TMSANetBackbone


class Model(RawBaselineModelBase):
    def __init__(self, params):
        super().__init__(params)
        num_classes = getattr(params, "num_of_classes", 2)
        num_channels = getattr(params, "num_channels", 64)
        samples = getattr(params, "samples", 1000)
        self.backbone = TMSANetBackbone(
            num_channels=num_channels,
            samples=samples,
            embed_dim=getattr(params, "embed_dim", 32),
            pool_size=getattr(params, "pool_size", 50),
            pool_stride=getattr(params, "pool_stride", 15),
            heads=getattr(params, "heads", 4),
            depth=getattr(params, "depth", 2),
            dropout=getattr(params, "dropout", 0.5),
        )
        self.classifier = LinearClassifierHead(self.backbone.feature_dim, num_classes)

    def extract_features(self, x):
        return self.backbone(x)
