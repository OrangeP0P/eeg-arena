"""
EEGPT Models Package
"""
from .eegpt_mcae import (
    CHANNEL_DICT,
    EEGTransformer,
    EEGTransformerReconstructor,
    EEGTransformerPredictor,
)
from .eegpt_mcae_finetune import (
    EEGPTClassifier,
    Conv1dWithConstraint,
    LinearWithConstraint,
    temporal_interpolation,
)

__all__ = [
    'CHANNEL_DICT',
    'EEGTransformer',
    'EEGTransformerReconstructor',
    'EEGTransformerPredictor',
    'EEGPTClassifier',
    'Conv1dWithConstraint',
    'LinearWithConstraint',
    'temporal_interpolation',
]
