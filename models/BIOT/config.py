"""
BIOT Model Config Class - Defines model-specific parameters

BIOT (Biosignal Transformer) converts EEG signals to "sentence" structure
and uses Transformer encoding in the frequency domain.
"""
import argparse
from dataclasses import dataclass


@dataclass
class BIOTDefaults:
    """
    BIOT default parameters - for IDE auto-completion
    """
    emb_size: int = 256
    heads: int = 8
    depth: int = 4
    dropout: float = 0.1
    n_fft: int = 200
    hop_length: int = 100


class BIOTConfig:
    """BIOT model configuration class"""

    # Model construction parameters (used to extract from args and pass to model)
    MODEL_KEYS = ['emb_size', 'heads', 'depth', 'dropout', 'n_fft', 'hop_length']

    # Pretrained weight variants
    WEIGHT_VARIANTS = {
        "shhs-prest-18ch": {
            "file": "EEG-SHHS+PREST-18-channels.ckpt",
            "description": "SHHS+PREST pretrained, 18 channels",
        },
        "prest-16ch": {
            "file": "EEG-PREST-16-channels.ckpt",
            "description": "PREST pretrained, 16 channels",
        },
        "six-datasets-18ch": {
            "file": "EEG-six-datasets-18-channels.ckpt",
            "description": "Six datasets pretrained, 18 channels",
        },
    }
    DEFAULT_VARIANT = "shhs-prest-18ch"
    WEIGHTS_DIR = "./models/Weights/BIOT"

    @staticmethod
    def get_model_keys():
        """Get the list of parameters required for model construction"""
        return BIOTConfig.MODEL_KEYS.copy()

    @staticmethod
    def get_input_constraints():
        """
        Model-level input constraints used by the training entry.
        BIOT uses raw EEG input (not patched).
        """
        return {
            "required_input_format": "raw",  # BIOT uses raw EEG
            "supports_pretrained": True,
        }

    @staticmethod
    def add_args(parser, mode='all'):
        """
        Add BIOT-specific arguments to argparse

        Args:
            parser: argparse.ArgumentParser instance
            mode: 'all', 'finetune', or 'pretrain'
        """
        if mode in ['all', 'finetune']:
            BIOTConfig._add_finetune_args(parser)

        if mode in ['all', 'pretrain']:
            BIOTConfig._add_pretrain_args(parser)

    @staticmethod
    def _add_finetune_args(parser):
        """Add fine-tune specific arguments"""
        group = parser.add_argument_group('BIOT Fine-tune options')

        # Model structure parameters
        group.add_argument(
            "--emb_size",
            type=int,
            default=256,
            help="Embedding size (default: 256)",
        )
        group.add_argument(
            "--heads",
            type=int,
            default=8,
            help="Number of attention heads (default: 8)",
        )
        group.add_argument(
            "--depth",
            type=int,
            default=4,
            help="Number of transformer layers (default: 4)",
        )
        group.add_argument(
            "--dropout",
            type=float,
            default=0.1,
            help="Dropout rate (default: 0.1)",
        )
        group.add_argument(
            "--n_fft",
            type=int,
            default=200,
            help="FFT size for STFT (default: 200)",
        )
        group.add_argument(
            "--hop_length",
            type=int,
            default=100,
            help="Hop length for STFT (default: 100)",
        )

        # Pretrained weight options
        group.add_argument(
            "--foundation_dir",
            type=str,
            default='./models/Weights/BIOT/EEG-SHHS+PREST-18-channels.ckpt',
            help="Path to pretrained weights (default: EEG-SHHS+PREST-18-channels.ckpt)",
        )
        group.add_argument(
            "--variant",
            type=str,
            default=None,
            choices=list(BIOTConfig.WEIGHT_VARIANTS.keys()),
            help="Pretrained weight variant: " +
                 ", ".join(f"{k} ({v['description']})" for k, v in BIOTConfig.WEIGHT_VARIANTS.items()),
        )

    @staticmethod
    def _add_pretrain_args(parser):
        """Add pre-train specific arguments"""
        group = parser.add_argument_group('BIOT Pre-train options')

        group.add_argument(
            "--emb_size",
            type=int,
            default=256,
            help="Embedding size (default: 256)",
        )
        group.add_argument(
            "--heads",
            type=int,
            default=8,
            help="Number of attention heads (default: 8)",
        )
        group.add_argument(
            "--depth",
            type=int,
            default=4,
            help="Number of transformer layers (default: 4)",
        )
        group.add_argument(
            "--n_fft",
            type=int,
            default=200,
            help="FFT size for STFT (default: 200)",
        )
        group.add_argument(
            "--hop_length",
            type=int,
            default=100,
            help="Hop length for STFT (default: 100)",
        )
