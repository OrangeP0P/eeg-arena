"""SlimSeiz config."""
from dataclasses import dataclass


@dataclass
class SlimSeizDefaults:
    hidden_dim: int = 96
    depth: int = 4
    dropout: float = 0.2


class SlimSeizConfig:
    MODEL_KEYS = ["hidden_dim", "depth", "dropout"]

    @staticmethod
    def get_model_keys():
        return SlimSeizConfig.MODEL_KEYS.copy()

    @staticmethod
    def get_input_constraints():
        return {"required_input_format": "raw", "supports_pretrained": False}

    @staticmethod
    def add_args(parser, mode="all"):
        if mode in ["all", "finetune"]:
            group = parser.add_argument_group("SlimSeiz Fine-tune options")
            group.add_argument("--hidden_dim", type=int, default=96, help="Hidden feature width (default: 96)")
            group.add_argument("--depth", type=int, default=4, help="Number of gated temporal blocks (default: 4)")
            group.add_argument("--dropout", type=float, default=0.2, help="Dropout rate (default: 0.2)")
