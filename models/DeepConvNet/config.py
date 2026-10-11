"""DeepConvNet config."""
from dataclasses import dataclass


@dataclass
class DeepConvNetDefaults:
    """DeepConvNet defaults for IDE hints."""

    n_filters_time: int = 25
    filter_time_length: int = 10
    pool_time_length: int = 3
    pool_time_stride: int = 3
    dropout: float = 0.5


class DeepConvNetConfig:
    """DeepConvNet model configuration."""

    MODEL_KEYS = [
        "n_filters_time",
        "filter_time_length",
        "pool_time_length",
        "pool_time_stride",
        "dropout",
    ]

    @staticmethod
    def get_model_keys():
        return DeepConvNetConfig.MODEL_KEYS.copy()

    @staticmethod
    def get_input_constraints():
        return {
            "required_input_format": "raw",
            "supports_pretrained": False,
        }

    @staticmethod
    def add_args(parser, mode="all"):
        if mode in ["all", "finetune"]:
            DeepConvNetConfig._add_finetune_args(parser)

    @staticmethod
    def _add_finetune_args(parser):
        group = parser.add_argument_group("DeepConvNet Fine-tune options")
        group.add_argument("--n_filters_time", type=int, default=25,
                           help="Base number of temporal filters (default: 25)")
        group.add_argument("--filter_time_length", type=int, default=10,
                           help="Temporal kernel length in each block (default: 10). "
                                "Automatically reduced if the input sequence is too short "
                                "for the given pooling configuration.")
        group.add_argument("--pool_time_length", type=int, default=3,
                           help="Temporal pooling length (default: 3)")
        group.add_argument("--pool_time_stride", type=int, default=3,
                           help="Temporal pooling stride (default: 3)")
        group.add_argument("--dropout", type=float, default=0.5,
                           help="Dropout rate (default: 0.5)")
