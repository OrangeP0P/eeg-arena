"""EEGNeX model components."""

from .backbone import EEGNeXBackbone
from .finetune_model import Model

__all__ = ["EEGNeXBackbone", "Model"]
