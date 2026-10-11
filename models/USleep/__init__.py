"""USleep model package."""
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data.finetune import FinetuneDataLoader
from models.registry import register_model

from .config import USleepConfig
from .models.finetune_model import Model as USleepFinetune

register_model(
    name="usleep",
    model_class=USleepFinetune,
    config_class=USleepConfig,
    dataset_adapter_class=FinetuneDataLoader,
)

__all__ = ["USleepConfig", "USleepFinetune"]
