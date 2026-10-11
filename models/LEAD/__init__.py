"""LEAD model package."""
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data.finetune import FinetuneDataLoader
from models.registry import register_model

from .config import LEADConfig
from .models.finetune_model import Model as LEADFinetune


register_model(
    name="lead",
    model_class=LEADFinetune,
    config_class=LEADConfig,
    dataset_adapter_class=FinetuneDataLoader,
)


__all__ = ["LEADConfig", "LEADFinetune"]
