"""ATCNet model package."""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.finetune import FinetuneDataLoader
from models.registry import register_model

from .config import ATCNetConfig
from .models.finetune_model import Model as ATCNetFinetune


register_model(
    name="atcnet",
    model_class=ATCNetFinetune,
    config_class=ATCNetConfig,
    dataset_adapter_class=FinetuneDataLoader,
)


__all__ = ["ATCNetConfig", "ATCNetFinetune"]
