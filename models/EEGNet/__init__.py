"""EEGNet model package."""
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models.registry import register_model
from .config import EEGNetConfig
from .models.finetune_model import Model as EEGNetFinetune
from data.finetune import FinetuneDataLoader

register_model(
    name="eegnet",
    model_class=EEGNetFinetune,
    config_class=EEGNetConfig,
    dataset_adapter_class=FinetuneDataLoader,
)

__all__ = ["EEGNetConfig", "EEGNetFinetune"]
