"""HEAR model package."""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.finetune import FinetuneDataLoader
from models.registry import register_model

from .config import HEARConfig
from .models.finetune_model import Model as HEARFinetune


register_model(
    name="hear",
    model_class=HEARFinetune,
    config_class=HEARConfig,
    dataset_adapter_class=FinetuneDataLoader,
)


__all__ = ["HEARConfig", "HEARFinetune"]
