"""MSVTNet config."""
import argparse
from dataclasses import dataclass


@dataclass
class MSVTNetDefaults:
    """MSVTNet defaults for IDE hints."""

    branch_filters: str = "9,9,9,9"
    branch_kernel_sizes: str = "15,31,63,125"
    temporal_kernel_size: int = 15
    depth_multiplier: int = 2
    pool_size1: int = 8
    pool_size2: int = 7
    cnn_dropout: float = 0.3
    attention_heads: int = 8
    ff_ratio: float = 1.0
    transformer_dropout: float = 0.5
    transformer_layers: int = 2
    enable_auxiliary_branch_loss: bool = True
    branch_loss_lambda: float = 0.6


class MSVTNetConfig:
    """MSVTNet model configuration."""

    MODEL_KEYS = [
        "branch_filters",
        "branch_kernel_sizes",
        "temporal_kernel_size",
        "depth_multiplier",
        "pool_size1",
        "pool_size2",
        "cnn_dropout",
        "attention_heads",
        "ff_ratio",
        "transformer_dropout",
        "transformer_layers",
        "enable_auxiliary_branch_loss",
        "branch_loss_lambda",
    ]

    @staticmethod
    def get_model_keys():
        return MSVTNetConfig.MODEL_KEYS.copy()

    @staticmethod
    def get_input_constraints():
        return {
            "required_input_format": "raw",
            "supports_pretrained": False,
        }

    @staticmethod
    def add_args(parser, mode="all"):
        if mode in ["all", "finetune"]:
            MSVTNetConfig._add_finetune_args(parser)

    @staticmethod
    def _add_finetune_args(parser):
        group = parser.add_argument_group("MSVTNet Fine-tune options")
        group.add_argument(
            "--branch_filters",
            type=str,
            default="9,9,9,9",
            help="Comma-separated temporal filters for each multi-scale branch (default: 9,9,9,9)",
        )
        group.add_argument(
            "--branch_kernel_sizes",
            type=str,
            default="15,31,63,125",
            help="Comma-separated temporal kernel sizes for the branch frontends (default: 15,31,63,125)",
        )
        group.add_argument(
            "--temporal_kernel_size",
            type=int,
            default=15,
            help="Kernel size of the depthwise temporal refinement conv in each branch (default: 15)",
        )
        group.add_argument(
            "--depth_multiplier",
            type=int,
            default=2,
            help="Depth multiplier used by the spatial depthwise conv in each branch (default: 2)",
        )
        group.add_argument(
            "--pool_size1",
            type=int,
            default=8,
            help="Temporal pooling size after the first branch block (default: 8)",
        )
        group.add_argument(
            "--pool_size2",
            type=int,
            default=7,
            help="Temporal pooling size after the second branch block (default: 7)",
        )
        group.add_argument(
            "--cnn_dropout",
            type=float,
            default=0.3,
            help="Dropout rate in the multi-scale convolution branches (default: 0.3)",
        )
        group.add_argument(
            "--attention_heads",
            type=int,
            default=8,
            help="Number of attention heads in the transformer encoder (default: 8)",
        )
        group.add_argument(
            "--ff_ratio",
            type=float,
            default=1.0,
            help="Feed-forward hidden width ratio relative to the transformer width (default: 1.0)",
        )
        group.add_argument(
            "--transformer_dropout",
            type=float,
            default=0.5,
            help="Dropout rate in the transformer encoder (default: 0.5)",
        )
        group.add_argument(
            "--transformer_layers",
            type=int,
            default=2,
            help="Number of transformer encoder layers (default: 2)",
        )
        group.add_argument(
            "--enable_auxiliary_branch_loss",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Enable the upstream-style auxiliary branch supervision during training (default: enabled)",
        )
        group.add_argument(
            "--branch_loss_lambda",
            type=float,
            default=0.6,
            help="Weight on the final-head loss in the joint MSVTNet loss (default: 0.6)",
        )
