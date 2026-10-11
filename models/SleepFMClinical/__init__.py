"""SleepFM Clinical integration for EEG-Arena."""

from data.finetune import FinetuneDataLoader
from models.registry import register_model

from .config import SleepFMClinicalConfig
from .models.finetune_model import Model


for _name in ("sleepfm-clinical", "sleepfmclinical"):
    register_model(
        name=_name,
        model_class=Model,
        config_class=SleepFMClinicalConfig,
        dataset_adapter_class=FinetuneDataLoader,
    )

__all__ = ["Model", "SleepFMClinicalConfig"]
