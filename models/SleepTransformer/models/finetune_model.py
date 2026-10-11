"""SleepTransformer fine-tune wrapper."""
from models._shared import LinearClassifierHead, RawBaselineModelBase, SleepTransformerBackbone


class Model(RawBaselineModelBase):
    def __init__(self, params):
        super().__init__(params)
        num_classes = getattr(params, "num_of_classes", 2)
        num_channels = getattr(params, "num_channels", 1)
        self.backbone = SleepTransformerBackbone(
            num_channels=num_channels,
            frame_size=getattr(params, "frame_size", 128),
            frame_step=getattr(params, "frame_step", 64),
            frame_d_model=getattr(params, "frame_d_model", 128),
            frame_heads=getattr(params, "frame_heads", 4),
            frame_depth=getattr(params, "frame_depth", 2),
            seq_d_model=getattr(params, "seq_d_model", 128),
            seq_heads=getattr(params, "seq_heads", 4),
            seq_depth=getattr(params, "seq_depth", 2),
            epoch_seq_len=getattr(params, "epoch_seq_len", 4),
            dropout=getattr(params, "dropout", 0.3),
        )
        self.classifier = LinearClassifierHead(self.backbone.feature_dim, num_classes)

    def extract_features(self, x):
        return self.backbone(x)
