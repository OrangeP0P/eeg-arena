"""BiDANN config."""
from dataclasses import dataclass


@dataclass
class BiDANNDefaults:
    segments: int = 9
    hidden_size: int = 128
    projected_steps: int = 8
    dropout: float = 0.3


class BiDANNConfig:
    MODEL_KEYS = ["segments", "hidden_size", "projected_steps", "dropout"]

    @staticmethod
    def get_model_keys():
        return BiDANNConfig.MODEL_KEYS.copy()

    @staticmethod
    def get_input_constraints():
        return {"required_input_format": "raw", "supports_pretrained": False}

    @staticmethod
    def add_args(parser, mode="all"):
        if mode in ["all", "finetune"]:
            group = parser.add_argument_group("BiDANN Fine-tune options")
            group.add_argument("--segments", type=int, default=9, help="Number of temporal segments per epoch (default: 9)")
            group.add_argument("--hidden_size", type=int, default=128, help="LSTM hidden size per hemisphere (default: 128)")
            group.add_argument("--projected_steps", type=int, default=8, help="Projected temporal steps after pooling (default: 8)")
            group.add_argument("--dropout", type=float, default=0.3, help="Dropout rate (default: 0.3)")
