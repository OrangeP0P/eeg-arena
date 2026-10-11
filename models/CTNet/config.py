"""CTNet config."""
from dataclasses import dataclass


@dataclass
class CTNetDefaults:
    """CTNet defaults for IDE hints."""

    heads: int = 4
    depth: int = 6
    emb_size: int = 40
    eeg_f1: int = 20
    eeg_d: int = 2
    eeg_kernel_size: int = 64
    eeg_temporal_kernel_size: int = 16
    eeg_pooling_size1: int = 8
    eeg_pooling_size2: int = 8
    eeg_dropout: float = 0.3
    attention_dropout: float = 0.5
    position_dropout: float = 0.1
    classifier_dropout: float = 0.5
    ffn_expansion: int = 4


class CTNetConfig:
    """CTNet model configuration."""

    MODEL_KEYS = [
        "heads",
        "depth",
        "emb_size",
        "eeg_f1",
        "eeg_d",
        "eeg_kernel_size",
        "eeg_temporal_kernel_size",
        "eeg_pooling_size1",
        "eeg_pooling_size2",
        "eeg_dropout",
        "attention_dropout",
        "position_dropout",
        "classifier_dropout",
        "ffn_expansion",
    ]

    @staticmethod
    def get_model_keys():
        return CTNetConfig.MODEL_KEYS.copy()

    @staticmethod
    def get_input_constraints():
        return {
            "required_input_format": "raw",
            "supports_pretrained": False,
        }

    @staticmethod
    def add_args(parser, mode="all"):
        if mode in ["all", "finetune"]:
            CTNetConfig._add_finetune_args(parser)

    @staticmethod
    def _add_finetune_args(parser):
        group = parser.add_argument_group("CTNet Fine-tune options")
        group.add_argument(
            "--heads",
            type=int,
            default=4,
            help="Number of transformer attention heads (default: 4)",
        )
        group.add_argument(
            "--depth",
            type=int,
            default=6,
            help="Number of transformer encoder blocks (default: 6)",
        )
        group.add_argument(
            "--emb_size",
            type=int,
            default=40,
            help="Transformer embedding size after the CNN frontend; set <= 0 to use eeg_f1 * eeg_d (default: 40)",
        )
        group.add_argument(
            "--eeg_f1",
            type=int,
            default=20,
            help="Number of temporal filters in the first convolution (default: 20)",
        )
        group.add_argument(
            "--eeg_d",
            type=int,
            default=2,
            help="Depth multiplier used by the spatial convolution block (default: 2)",
        )
        group.add_argument(
            "--eeg_kernel_size",
            type=int,
            default=64,
            help="Temporal kernel size in the first convolution block (default: 64)",
        )
        group.add_argument(
            "--eeg_temporal_kernel_size",
            type=int,
            default=16,
            help="Temporal kernel size in the second convolution block (default: 16)",
        )
        group.add_argument(
            "--eeg_pooling_size1",
            type=int,
            default=8,
            help="Temporal pooling size after the spatial convolution block (default: 8)",
        )
        group.add_argument(
            "--eeg_pooling_size2",
            type=int,
            default=8,
            help="Temporal pooling size before the transformer encoder (default: 8)",
        )
        group.add_argument(
            "--eeg_dropout",
            type=float,
            default=0.3,
            help="Dropout rate used by the CNN frontend (default: 0.3)",
        )
        group.add_argument(
            "--attention_dropout",
            type=float,
            default=0.5,
            help="Dropout rate used inside the transformer encoder blocks (default: 0.5)",
        )
        group.add_argument(
            "--position_dropout",
            type=float,
            default=0.1,
            help="Dropout rate applied after adding learnable positional embeddings (default: 0.1)",
        )
        group.add_argument(
            "--classifier_dropout",
            type=float,
            default=0.5,
            help="Dropout rate used by the linear classifier head (default: 0.5)",
        )
        group.add_argument(
            "--ffn_expansion",
            type=int,
            default=4,
            help="Expansion ratio of the transformer feed-forward hidden layer (default: 4)",
        )
