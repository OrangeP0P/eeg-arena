"""ADFCNN config."""
from dataclasses import dataclass


@dataclass
class ADFCNNDefaults:
    F1: int = 8
    D: int = 1
    dropout: float = 0.25


class ADFCNNConfig:
    MODEL_KEYS = ["F1", "D", "dropout"]

    @staticmethod
    def get_model_keys():
        return ADFCNNConfig.MODEL_KEYS.copy()

    @staticmethod
    def get_input_constraints():
        return {"required_input_format": "raw", "supports_pretrained": False}

    @staticmethod
    def add_args(parser, mode="all"):
        if mode in ["all", "finetune"]:
            group = parser.add_argument_group("ADFCNN Fine-tune options")
            group.add_argument("--F1", type=int, default=8, help="Base spectral filter count (default: 8)")
            group.add_argument("--D", type=int, default=1, help="Depth multiplier for the spatial branch (default: 1)")
            group.add_argument("--dropout", type=float, default=0.25, help="Dropout rate (default: 0.25)")
