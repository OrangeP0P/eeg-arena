"""CTNet fine-tuning wrapper for the unified EEG framework."""

import torch
import torch.nn as nn

from models._shared import RawBaselineModelBase

from .backbone import CTNetBackbone


class Model(RawBaselineModelBase):
    """Unified CTNet wrapper where `forward(x)` returns logits."""

    def __init__(self, params):
        super().__init__(params)
        num_classes = getattr(params, "num_of_classes", 2)
        num_channels = getattr(params, "num_channels", 64)
        samples = getattr(params, "samples", 1000)

        heads = getattr(params, "heads", 4)
        depth = getattr(params, "depth", 6)
        emb_size = getattr(params, "emb_size", 40)
        eeg_f1 = getattr(params, "eeg_f1", 20)
        eeg_d = getattr(params, "eeg_d", 2)
        eeg_kernel_size = getattr(params, "eeg_kernel_size", 64)
        eeg_temporal_kernel_size = getattr(params, "eeg_temporal_kernel_size", 16)
        eeg_pooling_size1 = getattr(params, "eeg_pooling_size1", 8)
        eeg_pooling_size2 = getattr(params, "eeg_pooling_size2", 8)
        eeg_dropout = getattr(params, "eeg_dropout", 0.3)
        attention_dropout = getattr(params, "attention_dropout", 0.5)
        position_dropout = getattr(params, "position_dropout", 0.1)
        classifier_dropout = getattr(params, "classifier_dropout", 0.5)
        ffn_expansion = getattr(params, "ffn_expansion", 4)

        self.backbone = CTNetBackbone(
            num_channels=num_channels,
            n_times=samples,
            heads=heads,
            depth=depth,
            emb_size=emb_size,
            eeg_f1=eeg_f1,
            eeg_d=eeg_d,
            eeg_kernel_size=eeg_kernel_size,
            eeg_temporal_kernel_size=eeg_temporal_kernel_size,
            eeg_pooling_size1=eeg_pooling_size1,
            eeg_pooling_size2=eeg_pooling_size2,
            eeg_dropout=eeg_dropout,
            attention_dropout=attention_dropout,
            position_dropout=position_dropout,
            ffn_expansion=ffn_expansion,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(self.backbone.feature_dim, num_classes),
        )

        if getattr(params, "frozen", False):
            self.freeze_backbone()

    def extract_features(self, x):
        if x.ndim != 3:
            raise ValueError(f"CTNet expects raw input [B, C, T], got shape {tuple(x.shape)}")
        features = self.backbone(x)
        return torch.flatten(features, start_dim=1)
