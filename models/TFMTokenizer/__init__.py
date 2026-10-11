"""Register TFM-Tokenizer/TFM-Encoder with EEG-Arena."""
from data.finetune import FinetuneDataLoader
from models.registry import register_model

from .config import TFMTokenizerConfig
from .models.finetune_model import Model


register_model(
    name="tfmtokenizer",
    model_class=Model,
    config_class=TFMTokenizerConfig,
    dataset_adapter_class=FinetuneDataLoader,
)

__all__ = ["Model", "TFMTokenizerConfig"]
