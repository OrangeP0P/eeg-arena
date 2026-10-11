"""EEGConformer config."""
from dataclasses import dataclass


@dataclass
class EEGConformerDefaults:
    """EEGConformer defaults for IDE hints."""

    n_filters_time: int = 40
    filter_time_length: int = 25
    pool_time_length: int = 75
    pool_time_stride: int = 15
    drop_prob: float = 0.5
    att_depth: int = 6
    att_heads: int = 10
    att_drop_prob: float = 0.5


class EEGConformerConfig:
    """EEGConformer model configuration."""

    MODEL_KEYS = [
        "n_filters_time",
        "filter_time_length",
        "pool_time_length",
        "pool_time_stride",
        "drop_prob",
        "att_depth",
        "att_heads",
        "att_drop_prob",
    ]

    @staticmethod
    def get_model_keys():
        return EEGConformerConfig.MODEL_KEYS.copy()

    @staticmethod
    def get_input_constraints():
        return {
            "required_input_format": "raw",
            "supports_pretrained": False,
        }

    @staticmethod
    def add_args(parser, mode="all"):
        if mode in ["all", "finetune"]:
            EEGConformerConfig._add_finetune_args(parser)

    @staticmethod
    def _add_finetune_args(parser):
        group = parser.add_argument_group("EEGConformer Fine-tune options")
        group.add_argument("--n_filters_time", type=int, default=40,
                           help="Number of temporal filters / embedding dim (default: 40)")
        group.add_argument("--filter_time_length", type=int, default=25,
                           help="Temporal filter length (default: 25)")
        group.add_argument("--pool_time_length", type=int, default=75,
                           help="Temporal pooling length (default: 75)")
        group.add_argument("--pool_time_stride", type=int, default=15,
                           help="Temporal pooling stride (default: 15)")
        group.add_argument("--drop_prob", type=float, default=0.5,
                           help="Patch embedding dropout (default: 0.5)")
        group.add_argument("--att_depth", type=int, default=6,
                           help="Transformer depth (default: 6)")
        group.add_argument("--att_heads", type=int, default=10,
                           help="Number of attention heads (default: 10)")
        group.add_argument("--att_drop_prob", type=float, default=0.5,
                           help="Attention dropout (default: 0.5)")
