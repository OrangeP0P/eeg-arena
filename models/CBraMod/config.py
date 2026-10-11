"""
CBraMod Model Config Class - Defines model-specific parameters
"""
import argparse
from dataclasses import dataclass


@dataclass
class CBraModPretrainDefaults:
    """
    CBraMod pretrain default parameters - for IDE auto-completion
    """
    in_dim: int = 200
    out_dim: int = 200
    d_model: int = 200
    dim_feedforward: int = 800
    patch_num: int = 30
    n_layer: int = 12
    nhead: int = 8

@dataclass
class CBraModFinetuneDefaults:
    """
    CBraMod finetune default parameters - for IDE auto-completion
    This dataclass is only for code hints, does not affect actual execution
    """
    d_model: int = 200
    dim_feedforward: int = 800
    n_layer: int = 12
    nhead: int = 8
    dropout: float = 0.1
    classifier: str = 'all_patch_reps'
    frozen: bool = False
    use_pretrained_weights: bool = True
    foundation_dir: str = './models/Weights/CBraMod/pretrained_weights.pth'

class CBraModConfig:
    """CBraMod model configuration class"""

    # Model construction parameters (used to extract from args and pass to model)
    MODEL_KEYS = ['in_dim', 'out_dim', 'd_model', 'dim_feedforward', 'patch_num', 'n_layer', 'nhead']

    @staticmethod
    def get_model_keys():
        """Get the list of parameters required for model construction"""
        return CBraModConfig.MODEL_KEYS.copy()

    @staticmethod
    def get_input_constraints():
        """
        Model-level input constraints used by the training entry.
        """
        return {
            "required_input_format": "patched",
            "required_patch_size": 200,
            "supports_pretrained": True,
        }

    @staticmethod
    def add_args(parser, mode='all'):
        """
        Add CBraMod-specific arguments to argparse

        Args:
            parser: argparse.ArgumentParser instance
            mode: 'all', 'finetune', or 'pretrain'
        """
        if mode in ['all', 'finetune']:
            CBraModConfig._add_finetune_args(parser)

        if mode in ['all', 'pretrain']:
            CBraModConfig._add_pretrain_args(parser)

    @staticmethod
    def _add_finetune_args(parser):
        """Add fine-tune specific arguments"""
        group = parser.add_argument_group('CBraMod Fine-tune options')

        # Model structure parameters
        group.add_argument(
            "--d_model",
            type=int,
            default=200,
            help="Model dimension (default: 200)",
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
            default=12,
            help="Number of transformer layers (default: 12)",
        )
        group.add_argument(
            "--dim_feedforward",
            type=int,
            default=800,
            help="Feedforward dimension (default: 800)",
        )

        # Classifier parameters
        group.add_argument(
            "--classifier",
            type=str,
            default='all_patch_reps',
            help="Classifier architecture type (default: all_patch_reps)",
        )
        group.add_argument(
            "--dropout",
            type=float,
            default=0.1,
            help="Dropout rate (default: 0.1)",
        )

        group.add_argument(
            "--foundation_dir",
            type=str,
            default='./models/Weights/CBraMod/pretrained_weights.pth',
            help="Path to pretrained weights",
        )
