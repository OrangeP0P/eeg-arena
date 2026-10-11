"""
BENDR Model Config Class - Defines model-specific parameters
"""
import argparse
from dataclasses import dataclass


@dataclass
class BENDRFinetuneDefaults:
    """
    BENDR finetune default parameters - for IDE auto-completion
    This dataclass is only for code hints, does not affect actual execution
    """
    encoder_h: int = 512
    contextualizer_hidden: int = 3076
    nhead: int = 8
    n_layer: int = 8
    dropout: float = 0.15
    frozen: bool = False
    use_pretrained_weights: bool = True
    foundation_dir: str = './models/Weights/BENDR'


class BENDRConfig:
    """BENDR model configuration class"""

    # Model construction parameters (used to extract from args and pass to model)
    MODEL_KEYS = ['encoder_h', 'contextualizer_hidden', 'nhead', 'n_layer', 'dropout']

    @staticmethod
    def get_model_keys():
        """Get the list of parameters required for model construction"""
        return BENDRConfig.MODEL_KEYS.copy()

    @staticmethod
    def get_input_constraints():
        """
        Model-level input constraints used by the training entry.
        BENDR requires raw EEG input (not patched).
        """
        return {
            "required_input_format": "raw",  # BENDR uses raw EEG input
            "supports_pretrained": True,      # BENDR supports pretrained weights
        }

    @staticmethod
    def add_args(parser, mode='all'):
        """
        Add BENDR-specific arguments to argparse

        Args:
            parser: argparse.ArgumentParser instance
            mode: 'all', 'finetune', or 'pretrain'
        """
        if mode in ['all', 'finetune']:
            BENDRConfig._add_finetune_args(parser)

    @staticmethod
    def _add_finetune_args(parser):
        """Add fine-tune specific arguments"""
        group = parser.add_argument_group('BENDR Fine-tune options')

        # Model structure parameters
        group.add_argument(
            "--encoder_h",
            type=int,
            default=512,
            help="Encoder hidden dimension (default: 512)",
        )
        group.add_argument(
            "--contextualizer_hidden",
            type=int,
            default=3076,
            help="Contextualizer feedforward hidden dimension (default: 3076)",
        )
        group.add_argument(
            "--nhead",
            type=int,
            default=8,
            help="Number of attention heads (default: 8)",
        )
        group.add_argument(
            "--n_layer",
            type=int,
            default=8,
            help="Number of transformer layers (default: 8)",
        )
        group.add_argument(
            "--dropout",
            type=float,
            default=0.15,
            help="Dropout rate (default: 0.15)",
        )

        # Pretraining and freezing options
        group.add_argument(
            "--foundation_dir",
            type=str,
            default='./models/Weights/BENDR',
            help="Path to pretrained weights directory",
        )
