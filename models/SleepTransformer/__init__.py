"""SleepTransformer model package."""
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data.finetune import FinetuneDataLoader
from models.registry import register_model

from .config import SleepTransformerConfig
from .models.finetune_model import Model as SleepTransformerFinetune

register_model(
    name="sleeptransformer",
    model_class=SleepTransformerFinetune,
    config_class=SleepTransformerConfig,
    dataset_adapter_class=FinetuneDataLoader,
)

__all__ = ["SleepTransformerConfig", "SleepTransformerFinetune"]
