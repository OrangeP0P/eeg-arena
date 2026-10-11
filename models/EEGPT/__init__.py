"""
EEGPT Model Package
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.registry import register_model
from .config import EEGPTConfig
from .models.finetune_model import Model as EEGPTFinetune
from data.finetune import FinetuneDataLoader

# Register EEGPT model (fine-tune version) - use underlying FinetuneDataLoader
register_model(
    name='eegpt',
    model_class=EEGPTFinetune,
    config_class=EEGPTConfig,
    dataset_adapter_class=FinetuneDataLoader,
)

__all__ = ['EEGPTFinetune', 'EEGPTConfig']
