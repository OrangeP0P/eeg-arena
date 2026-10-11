import torch.nn as nn

from .encoder import ClassificationHead, EEGConformerEncoder


class Model(nn.Module):
    """EEGConformer fine-tuning model using a local source-based implementation."""

    def __init__(self, params):
        super().__init__()
        self.params = params

        num_classes = getattr(params, "num_of_classes", 2)
        num_channels = getattr(params, "num_channels", 64)
        samples = getattr(params, "samples", 1000)
        n_filters_time = getattr(params, "n_filters_time", 40)
        filter_time_length = getattr(params, "filter_time_length", 25)
        pool_time_length = getattr(params, "pool_time_length", 75)
        pool_time_stride = getattr(params, "pool_time_stride", 15)
        drop_prob = getattr(params, "drop_prob", 0.5)
        att_depth = getattr(params, "att_depth", 6)
        att_heads = getattr(params, "att_heads", 10)
        att_drop_prob = getattr(params, "att_drop_prob", 0.5)

        self.backbone = EEGConformerEncoder(
            n_chans=num_channels,
            n_times=samples,
            n_filters_time=n_filters_time,
            filter_time_length=filter_time_length,
            pool_time_length=pool_time_length,
            pool_time_stride=pool_time_stride,
            drop_prob=drop_prob,
            att_depth=att_depth,
            att_heads=att_heads,
            att_drop_prob=att_drop_prob,
        )
        self.classifier = ClassificationHead(
            feature_dim=self.backbone.feature_dim,
            emb_size=n_filters_time,
            n_classes=num_classes,
        )

        if getattr(params, "frozen", False):
            self.freeze_backbone()

    def _validate_input(self, x):
        if x.ndim != 3:
            raise ValueError(f"EEGConformer expects raw input [B, C, T], got shape {tuple(x.shape)}")
        return x

    def forward(self, x):
        x = self._validate_input(x)
        tokens = self.backbone(x)
        return self.classifier(tokens)

    def extract_features(self, x):
        x = self._validate_input(x)
        tokens = self.backbone(x)
        return self.classifier.flatten_tokens(tokens)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def get_param_groups(self):
        encoder_lr_scale = getattr(self.params, "encoder_lr_scale", 1.0)
        classifier_lr_scale = getattr(self.params, "classifier_lr_scale", 1.0)
        return [
            {"params": list(self.backbone.parameters()), "lr_scale": encoder_lr_scale},
            {"params": list(self.classifier.parameters()), "lr_scale": classifier_lr_scale},
        ]
