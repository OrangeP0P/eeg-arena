"""CTNet model package."""
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data.finetune import FinetuneDataLoader
from models.registry import register_model

from .config import CTNetConfig
from .models.finetune_model import Model as CTNetFinetune


register_model(
    name="ctnet",
    model_class=CTNetFinetune,
    config_class=CTNetConfig,
    dataset_adapter_class=FinetuneDataLoader,
)


__all__ = ["CTNetConfig", "CTNetFinetune"]
