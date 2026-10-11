"""ShallowConvNet model exports."""

from .backbone import ShallowConvNetBackbone
from .finetune_model import Model

__all__ = ["Model", "ShallowConvNetBackbone"]
