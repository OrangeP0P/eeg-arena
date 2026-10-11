"""FBCNet config."""
from dataclasses import dataclass


@dataclass
class FBCNetDefaults:
    m: int = 16
    stat_segments: int = 4


class FBCNetConfig:
    MODEL_KEYS = ["m", "stat_segments"]

    @staticmethod
    def get_model_keys():
        return FBCNetConfig.MODEL_KEYS.copy()

    @staticmethod
    def get_input_constraints():
        return {"required_input_format": "raw", "supports_pretrained": False}

    @staticmethod
    def add_args(parser, mode="all"):
        if mode in ["all", "finetune"]:
            group = parser.add_argument_group("FBCNet Fine-tune options")
            group.add_argument("--m", type=int, default=16, help="Number of spatial filters per band (default: 16)")
            group.add_argument(
                "--stat_segments",
                type=int,
                default=4,
                help="Number of temporal statistic segments after filter-bank pooling (default: 4)",
            )
