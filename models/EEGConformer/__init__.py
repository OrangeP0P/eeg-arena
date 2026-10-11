"""EEGConformer model package."""
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models.registry import register_model
from .config import EEGConformerConfig
from .models.finetune_model import Model as EEGConformerFinetune
from data.finetune import FinetuneDataLoader

register_model(
    name="eegconformer",
    model_class=EEGConformerFinetune,
    config_class=EEGConformerConfig,
    dataset_adapter_class=FinetuneDataLoader,
)

__all__ = ["EEGConformerConfig", "EEGConformerFinetune"]
