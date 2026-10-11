"""TMSANet model package."""
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data.finetune import FinetuneDataLoader
from models.registry import register_model

from .config import TMSANetConfig
from .models.finetune_model import Model as TMSANetFinetune

register_model(
    name="tmsanet",
    model_class=TMSANetFinetune,
    config_class=TMSANetConfig,
    dataset_adapter_class=FinetuneDataLoader,
)

__all__ = ["TMSANetConfig", "TMSANetFinetune"]
