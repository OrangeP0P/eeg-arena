"""
NeuroGPT Model Config Class - Defines model-specific parameters
"""
import argparse
from dataclasses import dataclass


@dataclass
class NeuroGPTPretrainDefaults:
    """
    NeuroGPT pretrain default parameters - for IDE auto-completion
    """
    embed_dim: int = 768
    num_hidden_layers: int = 6
    num_attention_heads: int = 12
    intermediate_dim_factor: int = 4
    n_positions: int = 512
    dropout: float = 0.1
    encoder_n_filters_time: int = 40
    encoder_filter_time_length: int = 25
    encoder_pool_time_length: int = 75
    encoder_pool_time_stride: int = 15
    encoder_drop_prob: float = 0.5
    encoder_att_depth: int = 6
    encoder_att_heads: int = 10
    encoder_att_drop_prob: float = 0.5


@dataclass
class NeuroGPTFinetuneDefaults:
    """
    NeuroGPT finetune default parameters - for IDE auto-completion
    This dataclass is only for code hints, does not affect actual execution
    """
    embed_dim: int = 768
    num_hidden_layers: int = 6
    num_attention_heads: int = 12
    intermediate_dim_factor: int = 4
    n_positions: int = 512
    dropout: float = 0.1
    frozen: bool = False
    use_pretrained_weights: bool = True
    foundation_dir: str = './models/Weights/NeuroGPT/pytorch_model.bin'


class NeuroGPTConfig:
    """NeuroGPT model configuration class"""

    # Model construction parameters (used to extract from args and pass to model)
    MODEL_KEYS = [
        'embed_dim', 'num_hidden_layers', 'num_attention_heads',
        'intermediate_dim_factor', 'n_positions', 'dropout',
        'encoder_n_filters_time', 'encoder_filter_time_length',
        'encoder_pool_time_length', 'encoder_pool_time_stride',
        'encoder_drop_prob', 'encoder_att_depth', 'encoder_att_heads',
        'encoder_att_drop_prob'
    ]

    @staticmethod
    def get_model_keys():
        """Get the list of parameters required for model construction"""
        return NeuroGPTConfig.MODEL_KEYS.copy()

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
        Add NeuroGPT-specific arguments to argparse

        Args:
            parser: argparse.ArgumentParser instance
            mode: 'all', 'finetune', or 'pretrain'
        """
        if mode in ['all', 'finetune']:
            NeuroGPTConfig._add_finetune_args(parser)

        if mode in ['all', 'pretrain']:
            NeuroGPTConfig._add_pretrain_args(parser)

    @staticmethod
    def _add_finetune_args(parser):
        """Add fine-tune specific arguments"""
        group = parser.add_argument_group('NeuroGPT Fine-tune options')

        # GPT decoder parameters
        group.add_argument(
            "--embed_dim",
            type=int,
            default=768,
            help="Transformer embedding dimension (default: 768)",
        )
        group.add_argument(
            "--num_hidden_layers",
            type=int,
            default=6,
            help="Number of GPT layers (default: 6)",
        )
        group.add_argument(
            "--num_attention_heads",
            type=int,
            default=12,
            help="Number of attention heads (default: 12)",
        )
        group.add_argument(
            "--intermediate_dim_factor",
            type=int,
            default=4,
            help="Feedforward dimension factor (default: 4)",
        )
        group.add_argument(
            "--n_positions",
            type=int,
            default=512,
            help="Maximum position embeddings (default: 512)",
        )
        group.add_argument(
            "--dropout",
            type=float,
            default=0.1,
            help="Dropout rate (default: 0.1)",
        )

        # Encoder parameters
        group.add_argument(
            "--encoder_n_filters_time",
            type=int,
            default=40,
            help="Number of temporal filters (default: 40)",
        )
        group.add_argument(
            "--encoder_filter_time_length",
            type=int,
            default=25,
            help="Temporal filter length (default: 25)",
        )
        group.add_argument(
            "--encoder_pool_time_length",
            type=int,
            default=75,
            help="Temporal pooling length (default: 75)",
        )
        group.add_argument(
            "--encoder_pool_time_stride",
            type=int,
            default=15,
            help="Temporal pooling stride (default: 15)",
        )
        group.add_argument(
            "--encoder_drop_prob",
            type=float,
            default=0.5,
            help="Encoder dropout probability (default: 0.5)",
        )
        group.add_argument(
            "--encoder_att_depth",
            type=int,
            default=6,
            help="Encoder transformer depth (default: 6)",
        )
        group.add_argument(
            "--encoder_att_heads",
            type=int,
            default=10,
            help="Encoder attention heads (default: 10)",
        )
        group.add_argument(
            "--encoder_att_drop_prob",
            type=float,
            default=0.5,
            help="Encoder attention dropout (default: 0.5)",
        )

        # Training parameters
        group.add_argument(
            "--foundation_dir",
            type=str,
            default='./models/Weights/NeuroGPT/pytorch_model.bin',
            help="Path to pretrained weights",
        )

    @staticmethod
    def _add_pretrain_args(parser):
        """Add pre-train specific arguments"""
        group = parser.add_argument_group('NeuroGPT Pre-train options')

        # Same as finetune args for now
        # GPT decoder parameters
        group.add_argument(
            "--embed_dim",
            type=int,
            default=768,
            help="Transformer embedding dimension (default: 768)",
        )
        group.add_argument(
            "--num_hidden_layers",
            type=int,
            default=6,
            help="Number of GPT layers (default: 6)",
        )
        group.add_argument(
            "--num_attention_heads",
            type=int,
            default=12,
            help="Number of attention heads (default: 12)",
        )
        group.add_argument(
            "--intermediate_dim_factor",
            type=int,
            default=4,
            help="Feedforward dimension factor (default: 4)",
        )
        group.add_argument(
            "--n_positions",
            type=int,
            default=512,
            help="Maximum position embeddings (default: 512)",
        )
        group.add_argument(
            "--dropout",
            type=float,
            default=0.1,
            help="Dropout rate (default: 0.1)",
        )
        # Encoder parameters
        group.add_argument(
            "--encoder_n_filters_time",
            type=int,
            default=40,
            help="Number of temporal filters (default: 40)",
        )
        group.add_argument(
            "--encoder_filter_time_length",
            type=int,
            default=25,
            help="Temporal filter length (default: 25)",
        )
        group.add_argument(
            "--encoder_pool_time_length",
            type=int,
            default=75,
            help="Temporal pooling length (default: 75)",
        )
        group.add_argument(
            "--encoder_pool_time_stride",
            type=int,
            default=15,
            help="Temporal pooling stride (default: 15)",
        )
        group.add_argument(
            "--encoder_drop_prob",
            type=float,
            default=0.5,
            help="Encoder dropout probability (default: 0.5)",
        )
        group.add_argument(
            "--encoder_att_depth",
            type=int,
            default=6,
            help="Encoder transformer depth (default: 6)",
        )
        group.add_argument(
            "--encoder_att_heads",
            type=int,
            default=10,
            help="Encoder attention heads (default: 10)",
        )
        group.add_argument(
            "--encoder_att_drop_prob",
            type=float,
            default=0.5,
            help="Encoder attention dropout (default: 0.5)",
        )
