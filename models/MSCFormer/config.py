"""MSCFormer config."""
from dataclasses import dataclass


@dataclass
class MSCFormerDefaults:
    f1: int = 16
    pooling_size: int = 52
    heads: int = 8
    depth: int = 5
    dropout: float = 0.25


class MSCFormerConfig:
    MODEL_KEYS = ["f1", "pooling_size", "heads", "depth", "dropout"]

    @staticmethod
    def get_model_keys():
        return MSCFormerConfig.MODEL_KEYS.copy()

    @staticmethod
    def get_input_constraints():
        return {"required_input_format": "raw", "supports_pretrained": False}

    @staticmethod
    def add_args(parser, mode="all"):
        if mode in ["all", "finetune"]:
            group = parser.add_argument_group("MSCFormer Fine-tune options")
            group.add_argument("--f1", type=int, default=16, help="Per-branch convolution width (default: 16)")
            group.add_argument("--pooling_size", type=int, default=52, help="Temporal pooling size in the CNN stem (default: 52)")
            group.add_argument("--heads", type=int, default=8, help="Number of transformer heads (default: 8)")
            group.add_argument("--depth", type=int, default=5, help="Number of transformer blocks (default: 5)")
            group.add_argument("--dropout", type=float, default=0.25, help="Dropout rate (default: 0.25)")
