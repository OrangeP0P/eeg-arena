"""EEGNeX config."""
from dataclasses import dataclass


@dataclass
class EEGNeXDefaults:
    """EEGNeX defaults for IDE hints."""

    stem_filters: int = 8
    mid_filters: int = 32
    depth_multiplier: int = 2
    kernel_length: int = 64
    dilated_kernel_length: int = 16
    first_dilation: int = 2
    second_dilation: int = 4
    pool_size: int = 4
    dropout: float = 0.5


class EEGNeXConfig:
    """EEGNeX model configuration."""

    MODEL_KEYS = [
        "stem_filters",
        "mid_filters",
        "depth_multiplier",
        "kernel_length",
        "dilated_kernel_length",
        "first_dilation",
        "second_dilation",
        "pool_size",
        "dropout",
    ]

    @staticmethod
    def get_model_keys():
        return EEGNeXConfig.MODEL_KEYS.copy()

    @staticmethod
    def get_input_constraints():
        return {
            "required_input_format": "raw",
            "supports_pretrained": False,
        }

    @staticmethod
    def add_args(parser, mode="all"):
        if mode in ["all", "finetune"]:
            EEGNeXConfig._add_finetune_args(parser)

    @staticmethod
    def _add_finetune_args(parser):
        group = parser.add_argument_group("EEGNeX Fine-tune options")
        group.add_argument(
            "--stem_filters",
            type=int,
            default=8,
            help="Number of filters in the first temporal convolution and the last dilated block (default: 8)",
        )
        group.add_argument(
            "--mid_filters",
            type=int,
            default=32,
            help="Number of filters in the second temporal convolution and the first dilated block (default: 32)",
        )
        group.add_argument(
            "--depth_multiplier",
            type=int,
            default=2,
            help="Depth multiplier used by the spatial depthwise convolution (default: 2)",
        )
        group.add_argument(
            "--kernel_length",
            type=int,
            default=64,
            help="Temporal kernel length of the stem convolutions (default: 64)",
        )
        group.add_argument(
            "--dilated_kernel_length",
            type=int,
            default=16,
            help="Temporal kernel length of the dilated convolutions (default: 16)",
        )
        group.add_argument(
            "--first_dilation",
            type=int,
            default=2,
            help="Dilation rate of the first dilated temporal convolution (default: 2)",
        )
        group.add_argument(
            "--second_dilation",
            type=int,
            default=4,
            help="Dilation rate of the second dilated temporal convolution (default: 4)",
        )
        group.add_argument(
            "--pool_size",
            type=int,
            default=4,
            help="Temporal average-pooling size used in both pooling stages (default: 4)",
        )
        group.add_argument(
            "--dropout",
            type=float,
            default=0.5,
            help="Dropout rate (default: 0.5)",
        )
