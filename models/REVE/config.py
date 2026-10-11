"""
REVE Model Config Class - Defines model-specific parameters

REVE (Recurrent Vision EEG) is a state-of-the-art EEG model that uses
4D positional embeddings (x, y, z, t) and requires electrode coordinates.
"""
import argparse
from dataclasses import dataclass


@dataclass
class REVEDefaults:
    """
    REVE default parameters - for IDE auto-completion
    """
    model_size: str = "large"  # "base" or "large"
    frozen: bool = False
    use_pretrained_weights: bool = True
    foundation_dir: str = './models/Weights/REVE'


# Default configs for base and large models
REVE_BASE_CONFIG = {
    "embed_dim": 512,
    "depth": 12,
    "heads": 8,
    "head_dim": 64,
    "mlp_dim_ratio": 2.66,
    "use_geglu": True,
    "freqs": 4,
    "noise_ratio": 0.0025,
    "patch_size": 200,
    "patch_overlap": 20,
}

REVE_LARGE_CONFIG = {
    "embed_dim": 512,
    "depth": 22,
    "heads": 8,
    "head_dim": 64,
    "mlp_dim_ratio": 2.66,
    "use_geglu": True,
    "freqs": 4,
    "noise_ratio": 0.0025,
    "patch_size": 200,
    "patch_overlap": 20,
}


class REVEConfig:
    """REVE model configuration class"""

    # Model construction parameters (used to extract from args and pass to model)
    MODEL_KEYS = ['model_size']

    # Config presets
    BASE_CONFIG = REVE_BASE_CONFIG
    LARGE_CONFIG = REVE_LARGE_CONFIG

    # Pretrained weight variants
    WEIGHT_VARIANTS = {
        "base": {
            "file": "reve-base.safetensors",
            "description": "REVE base model (12 layers)",
        },
        "base-002pct": {
            "file": "reve-base-002pct.safetensors",
            "description": "REVE base model pretrained on 2% of the data",
        },
        "base-005pct": {
            "file": "reve-base-005pct.safetensors",
            "description": "REVE base model pretrained on 5% of the data",
        },
        "base-010pct": {
            "file": "reve-base-010pct.safetensors",
            "description": "REVE base model pretrained on 10% of the data",
        },
        "base-025pct": {
            "file": "reve-base-025pct.safetensors",
            "description": "REVE base model pretrained on 25% of the data",
        },
        "base-50pct": {
            "file": "reve-base-50pct.safetensors",
            "description": "REVE base model pretrained on 50% of the data",
        },
        "base-100pct": {
            "file": "reve-base-100pct.safetensors",
            "description": "REVE base model pretrained on 100% of the data",
        },
        "large": {
            "file": "reve-large.safetensors",
            "description": "REVE large model (22 layers)",
        },
    }
    DEFAULT_VARIANT = "base"
    WEIGHTS_DIR = "./models/Weights/REVE"

    @staticmethod
    def get_model_keys():
        """Get the list of parameters required for model construction"""
        return REVEConfig.MODEL_KEYS.copy()

    @staticmethod
    def get_input_constraints():
        """
        Model-level input constraints used by the training entry.
        REVE uses raw EEG input and requires electrode positions.
        """
        return {
            "required_input_format": "raw",  # REVE uses raw EEG and does internal patching
            "supports_pretrained": True,
        }

    @staticmethod
    def add_args(parser, mode='all'):
        """
        Add REVE-specific arguments to argparse

        Args:
            parser: argparse.ArgumentParser instance
            mode: 'all', 'finetune', or 'pretrain'
        """
        if mode in ['all', 'finetune']:
            REVEConfig._add_finetune_args(parser)

        if mode in ['all', 'pretrain']:
            REVEConfig._add_pretrain_args(parser)

    @staticmethod
    def _add_finetune_args(parser):
        """Add fine-tune specific arguments"""
        group = parser.add_argument_group('REVE Fine-tune options')

        # Model size
        group.add_argument(
            "--model_size",
            type=str,
            default="base",
            choices=["base", "large"],
            help="REVE model size (default: base)",
        )

        # Model structure parameters
        group.add_argument(
            "--embed_dim",
            type=int,
            default=512,
            help="Embedding dimension (default: 512)",
        )
        group.add_argument(
            "--depth",
            type=int,
            default=22,
            help="Number of transformer layers (default: 22)",
        )
        group.add_argument(
            "--heads",
            type=int,
            default=8,
            help="Number of attention heads (default: 8)",
        )
        group.add_argument(
            "--head_dim",
            type=int,
            default=64,
            help="Head dimension (default: 64)",
        )
        group.add_argument(
            "--mlp_dim_ratio",
            type=float,
            default=2.66,
            help="MLP dimension ratio (default: 2.66)",
        )
        group.add_argument(
            "--use_geglu",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Use GEGLU activation (default: True)",
        )
        group.add_argument(
            "--freqs",
            type=int,
            default=4,
            help="Number of Fourier frequencies (default: 4)",
        )
        group.add_argument(
            "--noise_ratio",
            type=float,
            default=0.0025,
            help="Noise ratio for preprocessing (default: 0.0025)",
        )
        group.add_argument(
            "--patch_overlap",
            type=int,
            default=20,
            help="Patch overlap size (default: 20)",
        )

        # Pretrained weight options
        group.add_argument(
            "--foundation_dir",
            type=str,
            default='./models/Weights/REVE',
            help="Path to pretrained weights directory (default: ./models/Weights/REVE)",
        )
        group.add_argument(
            "--variant",
            type=str,
            default=None,
            choices=list(REVEConfig.WEIGHT_VARIANTS.keys()),
            help="Pretrained weight variant: " +
                 ", ".join(f"{k} ({v['description']})" for k, v in REVEConfig.WEIGHT_VARIANTS.items()),
        )

    @staticmethod
    def _add_pretrain_args(parser):
        """Add pre-train specific arguments"""
        group = parser.add_argument_group('REVE Pre-train options')

        group.add_argument(
            "--embed_dim",
            type=int,
            default=512,
            help="Embedding dimension (default: 512)",
        )
        group.add_argument(
            "--depth",
            type=int,
            default=22,
            help="Number of transformer layers (default: 22)",
        )
        group.add_argument(
            "--heads",
            type=int,
            default=8,
            help="Number of attention heads (default: 8)",
        )
        group.add_argument(
            "--head_dim",
            type=int,
            default=64,
            help="Head dimension (default: 64)",
        )
        group.add_argument(
            "--mlp_dim_ratio",
            type=float,
            default=2.66,
            help="MLP dimension ratio (default: 2.66)",
        )
        group.add_argument(
            "--freqs",
            type=int,
            default=4,
            help="Number of Fourier frequencies (default: 4)",
        )
        group.add_argument(
            "--patch_size",
            type=int,
            default=200,
            help="Patch size for EEG segmentation (default: 200)",
        )
