"""
CSBrain Model Package
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.registry import register_model
from .config import CSBrainConfig
from .models.finetune_model import CSBrainFinetune
from data.finetune import FinetuneDataLoader

# Register CSBrain model (fine-tune version) - use underlying FinetuneDataLoader
register_model(
    name='csbrain',
    model_class=CSBrainFinetune,
    config_class=CSBrainConfig,
    dataset_adapter_class=FinetuneDataLoader,
)

__all__ = ['CSBrainFinetune', 'CSBrainConfig']
