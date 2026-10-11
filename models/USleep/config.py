"""USleep config."""
from dataclasses import dataclass


@dataclass
class USleepDefaults:
    base_filters: int = 16
    depth: int = 4
    classifier_hidden: int = 64
    dropout: float = 0.25


class USleepConfig:
    MODEL_KEYS = ["base_filters", "depth", "classifier_hidden", "dropout"]

    @staticmethod
    def get_model_keys():
        return USleepConfig.MODEL_KEYS.copy()

    @staticmethod
    def get_input_constraints():
        return {"required_input_format": "raw", "supports_pretrained": False}

    @staticmethod
    def add_args(parser, mode="all"):
        if mode in ["all", "finetune"]:
            group = parser.add_argument_group("USleep Fine-tune options")
            group.add_argument("--base_filters", type=int, default=16, help="Initial U-Net filter width (default: 16)")
            group.add_argument("--depth", type=int, default=4, help="Number of encoder/decoder levels (default: 4)")
            group.add_argument("--classifier_hidden", type=int, default=64, help="Hidden width of the classification head (default: 64)")
            group.add_argument("--dropout", type=float, default=0.25, help="Dropout rate in the classifier head (default: 0.25)")
