"""
EEGMamba model package.
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.finetune import FinetuneDataLoader
from models.registry import register_model

from .config import EEGMambaConfig
from .models.finetune_model import Model as EEGMambaFinetune


register_model(
    name="eegmamba",
    model_class=EEGMambaFinetune,
    config_class=EEGMambaConfig,
    dataset_adapter_class=FinetuneDataLoader,
)


__all__ = ["EEGMambaConfig", "EEGMambaFinetune"]
