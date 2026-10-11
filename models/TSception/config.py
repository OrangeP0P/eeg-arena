"""TSception config."""
from dataclasses import dataclass


@dataclass
class TSceptionDefaults:
    sampling_rate: int = 200
    num_T: int = 15
    num_S: int = 15
    classifier_hidden: int = 32
    dropout: float = 0.5


class TSceptionConfig:
    MODEL_KEYS = ["sampling_rate", "num_T", "num_S", "classifier_hidden", "dropout"]

    @staticmethod
    def get_model_keys():
        return TSceptionConfig.MODEL_KEYS.copy()

    @staticmethod
    def get_input_constraints():
        return {"required_input_format": "raw", "supports_pretrained": False}

    @staticmethod
    def add_args(parser, mode="all"):
        if mode in ["all", "finetune"]:
            group = parser.add_argument_group("TSception Fine-tune options")
            group.add_argument("--sampling_rate", type=int, default=200, help="Input sampling rate in Hz (default: 200)")
            group.add_argument("--num_T", type=int, default=15, help="Number of temporal filters per inception branch (default: 15)")
            group.add_argument("--num_S", type=int, default=15, help="Number of spatial filters per spatial branch (default: 15)")
            group.add_argument("--classifier_hidden", type=int, default=32, help="Hidden width of the classification head (default: 32)")
            group.add_argument("--dropout", type=float, default=0.5, help="Dropout rate in the classifier head (default: 0.5)")
