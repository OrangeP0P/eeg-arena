"""Register Uni-DMFM with EEG-Arena."""
from data.finetune import FinetuneDataLoader
from models.registry import register_model

from .config import UniDMFMConfig
from .models.finetune_model import Model


register_model(
    name="unidmfm",
    model_class=Model,
    config_class=UniDMFMConfig,
    dataset_adapter_class=FinetuneDataLoader,
)

__all__ = ["Model", "UniDMFMConfig"]
