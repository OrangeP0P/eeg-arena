"""YASA config."""
from dataclasses import dataclass


@dataclass
class YASADefaults:
    sampling_rate: int = 200
    hidden_dim: int = 128
    dropout: float = 0.2


class YASAConfig:
    MODEL_KEYS = ["sampling_rate", "hidden_dim", "dropout"]

    @staticmethod
    def get_model_keys():
        return YASAConfig.MODEL_KEYS.copy()

    @staticmethod
    def get_input_constraints():
        return {"required_input_format": "raw", "supports_pretrained": False}

    @staticmethod
    def add_args(parser, mode="all"):
        if mode in ["all", "finetune"]:
            group = parser.add_argument_group("YASA Fine-tune options")
            group.add_argument("--sampling_rate", type=int, default=200, help="Input sampling rate in Hz (default: 200)")
            group.add_argument("--hidden_dim", type=int, default=128, help="Feature projection width (default: 128)")
            group.add_argument("--dropout", type=float, default=0.2, help="Dropout rate (default: 0.2)")
