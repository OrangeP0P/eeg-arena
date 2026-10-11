"""DBConformer fine-tune wrapper."""
from models._shared import DBConformerBackbone, MLPClassifierHead, RawBaselineModelBase


class Model(RawBaselineModelBase):
    def __init__(self, params):
        super().__init__(params)
        num_classes = getattr(params, "num_of_classes", 2)
        num_channels = getattr(params, "num_channels", 64)
        samples = getattr(params, "samples", 1000)
        dropout = getattr(params, "dropout", 0.5)

        self.backbone = DBConformerBackbone(
            num_channels=num_channels,
            samples=samples,
            emb_size=getattr(params, "emb_size", 40),
            tem_depth=getattr(params, "tem_depth", 4),
            spa_depth=getattr(params, "spa_depth", 4),
            radix=getattr(params, "radix", 2),
            kernel_size=getattr(params, "kernel_size", 63),
            temporal_patch_size=getattr(params, "temporal_patch_size", 125),
            spa_dim=getattr(params, "spa_dim", 32),
            dropout=dropout,
        )
        self.classifier = MLPClassifierHead(self.backbone.feature_dim, 64, num_classes, dropout=dropout)

    def extract_features(self, x):
        return self.backbone(x)
