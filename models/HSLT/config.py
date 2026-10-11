"""HSLT config."""
from dataclasses import dataclass


@dataclass
class HSLTDefaults:
    feature_bins: int = 5
    De: int = 8
    Dr: int = 16
    Le: int = 2
    Lr: int = 2


class HSLTConfig:
    MODEL_KEYS = ["feature_bins", "De", "Dr", "Le", "Lr"]

    @staticmethod
    def get_model_keys():
        return HSLTConfig.MODEL_KEYS.copy()

    @staticmethod
    def get_input_constraints():
        return {"required_input_format": "raw", "supports_pretrained": False}

    @staticmethod
    def add_args(parser, mode="all"):
        if mode in ["all", "finetune"]:
            group = parser.add_argument_group("HSLT Fine-tune options")
            group.add_argument("--feature_bins", type=int, default=5, help="Per-channel temporal summary bins (default: 5)")
            group.add_argument("--De", type=int, default=8, help="Electrode-level embedding size (default: 8)")
            group.add_argument("--Dr", type=int, default=16, help="Region-level embedding size (default: 16)")
            group.add_argument("--Le", type=int, default=2, help="Electrode-level transformer depth (default: 2)")
            group.add_argument("--Lr", type=int, default=2, help="Region-level transformer depth (default: 2)")
