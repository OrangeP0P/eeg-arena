"""
REVE Model Package

REVE (Recurrent Vision EEG) is a state-of-the-art EEG model that uses
4D positional embeddings (x, y, z, t) and requires electrode coordinates.
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.registry import register_model
from .config import REVEConfig
from .models.finetune_model import Model as REVEFinetune
from data.finetune import FinetuneDataLoader


# Register REVE model (fine-tune version) - uses standard FinetuneDataLoader
register_model(
    name='reve',
    model_class=REVEFinetune,
    config_class=REVEConfig,
    dataset_adapter_class=FinetuneDataLoader,  # Standard adapter - ch_positions will be in params
)

__all__ = ['REVEFinetune', 'REVEConfig', 'FinetuneDataLoader']
