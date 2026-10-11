"""
NeuroGPT Model Package
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.registry import register_model
from .config import NeuroGPTConfig
from .models.finetune_model import Model as NeuroGPTFinetune
from data.finetune import FinetuneDataLoader

# Register NeuroGPT model (fine-tune version) - use underlying FinetuneDataLoader
register_model(
    name='neurogpt',
    model_class=NeuroGPTFinetune,
    config_class=NeuroGPTConfig,
    dataset_adapter_class=FinetuneDataLoader,
)

__all__ = ['NeuroGPTFinetune', 'NeuroGPTConfig']
