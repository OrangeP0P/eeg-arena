"""
EEGPT Model Config Class - Defines model-specific parameters
"""
import argparse
from dataclasses import dataclass


@dataclass
class EEGPTFinetuneDefaults:
    """
    EEGPT finetune default parameters - for IDE auto-completion
    This dataclass is only for code hints, does not affect actual execution
    """
    embed_dim: int = 512
    depth: int = 8
    num_heads: int = 8
    patch_size: int = 64
    patch_stride: int = 64
    embed_num: int = 4
    mlp_ratio: float = 4.0
    use_pretrained_weights: bool = True
    foundation_dir: str = './models/Weights/EEGPT/eegpt_mcae_58chs_4s_large4E.ckpt'
    frozen: bool = False
    use_mean_pooling: bool = True
    use_chan_conv: bool = True  # Enable by default for flexible channel counts
    use_predictor: bool = False
    use_out_proj: bool = False
    desired_time_len: int = 1024  # Default to 4s at 256Hz


class EEGPTConfig:
    """EEGPT model configuration class"""

    # Model construction parameters (used to extract from args and pass to model)
    # Note: patch_size is handled by common args
    MODEL_KEYS = ['embed_dim', 'depth', 'num_heads', 'patch_stride',
                  'embed_num', 'mlp_ratio', 'use_pretrained_weights', 'foundation_dir',
                  'frozen', 'use_mean_pooling', 'use_chan_conv', 'use_predictor', 'use_out_proj',
                  'desired_time_len']

    @staticmethod
    def get_model_keys():
        """Get the list of parameters required for model construction"""
        return EEGPTConfig.MODEL_KEYS.copy()

    @staticmethod
    def get_input_constraints():
        """
        Model-level input constraints used by the training entry.
        """
        return {
            "required_input_format": "patched",
            "required_patch_size": 64,
            "supports_pretrained": True,
        }

    @staticmethod
    def add_args(parser, mode='all'):
        """
        Add EEGPT-specific arguments to argparse

        Args:
            parser: argparse.ArgumentParser instance
            mode: 'all', 'finetune', or 'pretrain'
        """
        if mode in ['all', 'finetune']:
            EEGPTConfig._add_finetune_args(parser)

    @staticmethod
    def _add_finetune_args(parser):
        """Add fine-tune specific arguments"""
        group = parser.add_argument_group('EEGPT Fine-tune options')

        # Model structure parameters (excluding patch_size as it's already in common args)
        group.add_argument(
            "--embed_dim",
            type=int,
            default=512,
            help="Embedding dimension (default: 512)",
        )
        group.add_argument(
            "--num_heads",
            type=int,
            default=8,
            help="Number of attention heads (default: 8)",
        )
        group.add_argument(
            "--depth",
            type=int,
            default=8,
            help="Number of transformer layers (default: 8)",
        )
        group.add_argument(
            "--patch_stride",
            type=int,
            default=64,
            help="Patch stride (default: 64)",
        )
        group.add_argument(
            "--embed_num",
            type=int,
            default=4,
            help="Number of embedding tokens (default: 4)",
        )
        group.add_argument(
            "--mlp_ratio",
            type=float,
            default=4.0,
            help="MLP expansion ratio (default: 4.0)",
        )

        # Classifier parameters
        group.add_argument(
            "--use_mean_pooling",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Use mean pooling for classification (default: True)",
        )
        group.add_argument(
            "--use_chan_conv",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Use channel convolution for flexible channel counts (default: True)",
        )
        group.add_argument(
            "--use_predictor",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Use predictor instead of reconstructor (default: False)",
        )
        group.add_argument(
            "--use_out_proj",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Use output projection (default: False)",
        )

        group.add_argument(
            "--foundation_dir",
            type=str,
            default='./models/Weights/EEGPT/eegpt_mcae_58chs_4s_large4E.ckpt',
            help="Path to pretrained weights",

        )
        group.add_argument(
            "--desired_time_len",
            type=int,
            default=1024,
            help="Desired time length for temporal interpolation (default: 1024)",
        )
