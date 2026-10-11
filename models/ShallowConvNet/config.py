"""ShallowConvNet config."""
from dataclasses import dataclass


@dataclass
class ShallowConvNetDefaults:
    """ShallowConvNet defaults for IDE hints."""

    n_filters_time: int = 40
    n_filters_spat: int = 40
    filter_time_length: int = 25
    pool_time_length: int = 75
    pool_time_stride: int = 15
    dropout: float = 0.5


class ShallowConvNetConfig:
    """ShallowConvNet model configuration."""

    MODEL_KEYS = [
        "n_filters_time",
        "n_filters_spat",
        "filter_time_length",
        "pool_time_length",
        "pool_time_stride",
        "dropout",
    ]

    @staticmethod
    def get_model_keys():
        return ShallowConvNetConfig.MODEL_KEYS.copy()

    @staticmethod
    def get_input_constraints():
        return {
            "required_input_format": "raw",
            "supports_pretrained": False,
        }

    @staticmethod
    def add_args(parser, mode="all"):
        if mode in ["all", "finetune"]:
            ShallowConvNetConfig._add_finetune_args(parser)

    @staticmethod
    def _add_finetune_args(parser):
        group = parser.add_argument_group("ShallowConvNet Fine-tune options")
        group.add_argument(
            "--n_filters_time",
            type=int,
            default=40,
            help="Number of temporal filters in the first layer (default: 40)",
        )
        group.add_argument(
            "--n_filters_spat",
            type=int,
            default=40,
            help="Number of spatial filters after channel mixing (default: 40)",
        )
        group.add_argument(
            "--filter_time_length",
            type=int,
            default=25,
            help="Temporal kernel length in the first layer (default: 25)",
        )
        group.add_argument(
            "--pool_time_length",
            type=int,
            default=75,
            help="Temporal average-pooling length (default: 75)",
        )
        group.add_argument(
            "--pool_time_stride",
            type=int,
            default=15,
            help="Temporal average-pooling stride (default: 15)",
        )
        group.add_argument(
            "--dropout",
            type=float,
            default=0.5,
            help="Dropout rate (default: 0.5)",
        )
