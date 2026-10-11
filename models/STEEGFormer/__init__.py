"""STEEGFormer model package."""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.finetune import FinetuneDataLoader
from models.registry import register_model

from .config import STEEGFormerConfig
from .models.finetune_model import Model as STEEGFormerFinetune


register_model(
    name="steegformer",
    model_class=STEEGFormerFinetune,
    config_class=STEEGFormerConfig,
    dataset_adapter_class=FinetuneDataLoader,
)


__all__ = ["STEEGFormerConfig", "STEEGFormerFinetune"]
