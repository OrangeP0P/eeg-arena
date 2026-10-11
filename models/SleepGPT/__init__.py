"""SleepGPT integration for the EEG-Arena unified fine-tuning interface."""

from data.finetune import FinetuneDataLoader
from models.registry import register_model

from .config import SleepGPTConfig
from .models.finetune_model import Model


register_model(
    name="sleepgpt",
    model_class=Model,
    config_class=SleepGPTConfig,
    dataset_adapter_class=FinetuneDataLoader,
)

__all__ = ["Model", "SleepGPTConfig"]
