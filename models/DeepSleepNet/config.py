"""DeepSleepNet config."""
from dataclasses import dataclass


@dataclass
class DeepSleepNetDefaults:
    target_steps: int = 8
    hidden_size: int = 128
    dropout: float = 0.5


class DeepSleepNetConfig:
    MODEL_KEYS = ["target_steps", "hidden_size", "dropout"]

    @staticmethod
    def get_model_keys():
        return DeepSleepNetConfig.MODEL_KEYS.copy()

    @staticmethod
    def get_input_constraints():
        return {"required_input_format": "raw", "supports_pretrained": False}

    @staticmethod
    def add_args(parser, mode="all"):
        if mode in ["all", "finetune"]:
            group = parser.add_argument_group("DeepSleepNet Fine-tune options")
            group.add_argument("--target_steps", type=int, default=8, help="Number of intra-epoch steps for the recurrent head (default: 8)")
            group.add_argument("--hidden_size", type=int, default=128, help="BiGRU hidden size (default: 128)")
            group.add_argument("--dropout", type=float, default=0.5, help="Dropout rate (default: 0.5)")
