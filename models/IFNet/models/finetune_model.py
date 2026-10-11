"""IFNet fine-tune wrapper."""
from models._shared import IFNetBackbone, LinearWithConstraint, RawBaselineModelBase


class Model(RawBaselineModelBase):
    def __init__(self, params):
        super().__init__(params)
        num_classes = getattr(params, "num_of_classes", 2)
        num_channels = getattr(params, "num_channels", 64)
        samples = getattr(params, "samples", 1000)
        self.backbone = IFNetBackbone(
            num_channels=num_channels,
            samples=samples,
            out_planes=getattr(params, "out_planes", 64),
            radix=getattr(params, "radix", 3),
            kernel_size=getattr(params, "kernel_size", 63),
            patch_size=getattr(params, "ifnet_patch_size", 125),
            dropout=getattr(params, "dropout", 0.5),
        )
        self.classifier = LinearWithConstraint(self.backbone.feature_dim, num_classes, max_norm=0.5)

    def extract_features(self, x):
        return self.backbone(x)
