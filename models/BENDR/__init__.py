"""
BENDR Model Package
"""
import sys
import os
import os.path

# Add project root to path for model_registry imports
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models.registry import register_model
from .config import BENDRConfig
from .models.finetune_model import Model as BENDRFinetune
from data.finetune import FinetuneDataLoader

# Register BENDR model (fine-tune version) - use underlying FinetuneDataLoader
register_model(
    name='bendr',
    model_class=BENDRFinetune,
    config_class=BENDRConfig,
    dataset_adapter_class=FinetuneDataLoader,
)

__all__ = ['BENDRFinetune', 'BENDRConfig']
