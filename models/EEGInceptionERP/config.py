"""EEGInceptionERP config."""
from dataclasses import dataclass


@dataclass
class EEGInceptionERPDefaults:
    """EEGInceptionERP defaults for IDE hints."""

    filters_per_branch: int = 8
    depth_multiplier: int = 2
    kernel_length_long: int = 64
    kernel_length_medium: int = 32
    kernel_length_short: int = 16
    dropout: float = 0.25
    activation: str = "elu"


class EEGInceptionERPConfig:
    """EEGInceptionERP model configuration."""

    MODEL_KEYS = [
        "filters_per_branch",
        "depth_multiplier",
        "kernel_length_long",
        "kernel_length_medium",
        "kernel_length_short",
        "dropout",
        "activation",
    ]

    @staticmethod
    def get_model_keys():
        return EEGInceptionERPConfig.MODEL_KEYS.copy()

    @staticmethod
    def get_input_constraints():
        return {
            "required_input_format": "raw",
            "supports_pretrained": False,
        }

    @staticmethod
    def add_args(parser, mode="all"):
        if mode in ["all", "finetune"]:
            EEGInceptionERPConfig._add_finetune_args(parser)

    @staticmethod
    def _add_finetune_args(parser):
        group = parser.add_argument_group("EEGInceptionERP Fine-tune options")
        group.add_argument(
            "--filters_per_branch",
            type=int,
            default=8,
            help="Number of temporal filters per Inception branch (default: 8)",
        )
        group.add_argument(
            "--depth_multiplier",
            type=int,
            default=2,
            help="Depth multiplier of the spatial depthwise convolution in block 1 (default: 2)",
        )
        group.add_argument(
            "--kernel_length_long",
            type=int,
            default=64,
            help="Longest temporal kernel length in samples (default: 64)",
        )
        group.add_argument(
            "--kernel_length_medium",
            type=int,
            default=32,
            help="Medium temporal kernel length in samples (default: 32)",
        )
        group.add_argument(
            "--kernel_length_short",
            type=int,
            default=16,
            help="Shortest temporal kernel length in samples (default: 16)",
        )
        group.add_argument(
            "--dropout",
            type=float,
            default=0.25,
            help="Dropout rate used throughout the network (default: 0.25)",
        )
        group.add_argument(
            "--activation",
            type=str,
            default="elu",
            choices=["elu", "relu", "gelu", "silu"],
            help="Activation used after each convolution block (default: elu)",
        )
