"""
LaBraM Model Config Class - Defines model-specific parameters
"""
import argparse
from dataclasses import dataclass


@dataclass
class LaBraMFinetuneDefaults:
    """
    LaBraM finetune default parameters - for IDE auto-completion
    This dataclass is only for code hints, does not affect actual execution
    """
    dropout: float = 0.0
    frozen: bool = False
    use_pretrained_weights: bool = True
    foundation_dir: str = './models/Weights/LaBraM'


class LaBraMConfig:
    """LaBraM model configuration class"""

    # Model construction parameters (used to extract from args and pass to model)
    # Note: LaBraM uses base model defaults; dropout is the main adjustable parameter
    MODEL_KEYS = ['dropout']

    @staticmethod
    def get_model_keys():
        """Get the list of parameters required for model construction"""
        return LaBraMConfig.MODEL_KEYS.copy()

    @staticmethod
    def get_input_constraints():
        """
        Model-level input constraints used by the training entry.
        LaBraM requires patched EEG input with patch_size=200.
        """
        return {
            "required_input_format": "patched",  # LaBraM uses patched EEG input
            "required_patch_size": 200,           # patch_size must be 200 for LaBraM
            "supports_pretrained": True,
        }

    @staticmethod
    def add_args(parser, mode='all'):
        """
        Add LaBraM-specific arguments to argparse

        Args:
            parser: argparse.ArgumentParser instance
            mode: 'all', 'finetune', or 'pretrain'
        """
        if mode in ['all', 'finetune']:
            LaBraMConfig._add_finetune_args(parser)

    @staticmethod
    def _add_finetune_args(parser):
        """Add fine-tune specific arguments"""
        group = parser.add_argument_group('LaBraM Fine-tune options')

        # Dropout parameter
        group.add_argument(
            "--dropout",
            type=float,
            default=0.0,
            help="Dropout rate (default: 0.0)",
        )

        # Pretraining and freezing options
        group.add_argument(
            "--foundation_dir",
            type=str,
            default='./models/Weights/LaBraM',
            help="Path to pretrained weights directory",
        )
