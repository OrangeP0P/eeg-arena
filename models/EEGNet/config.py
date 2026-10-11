"""EEGNet config."""
from dataclasses import dataclass


@dataclass
class EEGNetDefaults:
    """EEGNet defaults for IDE hints."""

    F1: int = 8
    D: int = 2
    kernel_length: int = 64
    dropout: float = 0.25


class EEGNetConfig:
    """EEGNet model configuration."""

    MODEL_KEYS = ["F1", "D", "kernel_length", "dropout"]

    @staticmethod
    def get_model_keys():
        return EEGNetConfig.MODEL_KEYS.copy()

    @staticmethod
    def get_input_constraints():
        return {
            "required_input_format": "raw",
            "supports_pretrained": False,
        }

    @staticmethod
    def add_args(parser, mode="all"):
        if mode in ["all", "finetune"]:
            EEGNetConfig._add_finetune_args(parser)

    @staticmethod
    def _add_finetune_args(parser):
        group = parser.add_argument_group("EEGNet Fine-tune options")
        group.add_argument("--F1", dest="F1", type=int, default=8,
                           help="Number of temporal filters in the first block (default: 8)")
        group.add_argument("--D", dest="D", type=int, default=2,
                           help="Depth multiplier for spatial filters (default: 2)")
        group.add_argument("--kernel_length", type=int, default=64,
                           help="Temporal kernel length in the first block (default: 64)")
        group.add_argument("--dropout", type=float, default=0.25,
                           help="Dropout rate (default: 0.25)")
