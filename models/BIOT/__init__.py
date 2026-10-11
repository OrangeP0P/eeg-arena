"""
BIOT Model Package

BIOT (Biosignal Transformer) converts EEG signals to "sentence" structure
and uses Transformer encoding in the frequency domain via STFT.
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.registry import register_model
from .config import BIOTConfig
from .models.finetune_model import Model as BIOTFinetune
from data.finetune import FinetuneDataLoader

# Register BIOT model (fine-tune version)
register_model(
    name='biot',
    model_class=BIOTFinetune,
    config_class=BIOTConfig,
    dataset_adapter_class=FinetuneDataLoader,
)

__all__ = ['BIOTFinetune', 'BIOTConfig']
