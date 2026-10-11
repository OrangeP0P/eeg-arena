"""
LaBraM Model Package
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.registry import register_model
from .config import LaBraMConfig
from .models.finetune_model import Model as LaBraMFinetune
from data.finetune import FinetuneDataLoader

# Register LaBraM model (fine-tune version) - use underlying FinetuneDataLoader
register_model(
    name='labram',
    model_class=LaBraMFinetune,
    config_class=LaBraMConfig,
    dataset_adapter_class=FinetuneDataLoader,
)

__all__ = ['LaBraMFinetune', 'LaBraMConfig']
