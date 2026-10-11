import torch.nn as nn


class RawBaselineModelBase(nn.Module):
    """Shared wrapper behavior for raw-input supervised baselines."""

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.backbone = None
        self.classifier = None

    def extract_features(self, x):
        raise NotImplementedError

    def forward(self, x):
        features = self.extract_features(x)
        return self.classifier(features)

    def freeze_backbone(self):
        if self.backbone is None:
            return
        for param in self.backbone.parameters():
            param.requires_grad = False

    def get_param_groups(self):
        encoder_lr_scale = getattr(self.params, "encoder_lr_scale", 1.0)
        classifier_lr_scale = getattr(self.params, "classifier_lr_scale", 10.0)
        backbone_params = list(self.backbone.parameters()) if self.backbone is not None else []
        classifier_params = list(self.classifier.parameters()) if self.classifier is not None else []
        return [
            {"params": backbone_params, "lr_scale": encoder_lr_scale},
            {"params": classifier_params, "lr_scale": classifier_lr_scale},
        ]
