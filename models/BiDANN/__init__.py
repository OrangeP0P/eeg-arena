"""BiDANN model package."""
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data.finetune import FinetuneDataLoader
from models.registry import register_model

from .config import BiDANNConfig
from .models.finetune_model import Model as BiDANNFinetune

register_model(
    name="bidann",
    model_class=BiDANNFinetune,
    config_class=BiDANNConfig,
    dataset_adapter_class=FinetuneDataLoader,
)

__all__ = ["BiDANNConfig", "BiDANNFinetune"]
