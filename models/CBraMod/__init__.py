"""
CBraMod Model Package
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.registry import register_model
from .config import CBraModConfig
from .models.finetune_model import Model as CBraModFinetune
from data.finetune import FinetuneDataLoader

# Register CBraMod model (fine-tune version) - use underlying FinetuneDataLoader
register_model(
    name='cbramod',
    model_class=CBraModFinetune,
    config_class=CBraModConfig,
    dataset_adapter_class=FinetuneDataLoader,
)

__all__ = ['CBraModFinetune', 'CBraModConfig']
