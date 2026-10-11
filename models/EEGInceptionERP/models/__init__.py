"""EEGInceptionERP model components."""

from .backbone import EEGInceptionERPBackbone
from .finetune_model import Model


__all__ = ["EEGInceptionERPBackbone", "Model"]
