"""Uni-DMFM configuration and released checkpoint catalogue."""
from dataclasses import dataclass


@dataclass
class UniDMFMFinetuneDefaults:
    foundation_dir: str = "./models/Weights/UniDMFM/pretrained_weights.pth"


class UniDMFMConfig:
    MODEL_KEYS = []
    WEIGHTS_DIR = "./models/Weights/UniDMFM"
    WEIGHT_VARIANTS = {
        "base": {
            "file": "pretrained_weights.pth",
            "description": "Official Uni-DMFM TUEG-pretrained checkpoint",
        },
    }
    DEFAULT_VARIANT = "base"
    DEFAULT_WEIGHT_PATH = "./models/Weights/UniDMFM/pretrained_weights.pth"

    @staticmethod
    def get_model_keys():
        return UniDMFMConfig.MODEL_KEYS.copy()

    @staticmethod
    def get_input_constraints():
        return {
            "required_input_format": "patched",
            "required_patch_size": 200,
            "supports_pretrained": True,
        }

    @staticmethod
    def add_args(parser, mode="all"):
        if mode not in {"all", "finetune"}:
            return
        group = parser.add_argument_group("Uni-DMFM fine-tune options")
        group.add_argument(
            "--foundation_dir",
            default=UniDMFMConfig.DEFAULT_WEIGHT_PATH,
            help="Official Uni-DMFM checkpoint file or weights directory.",
        )
        group.add_argument(
            "--variant",
            choices=list(UniDMFMConfig.WEIGHT_VARIANTS),
            default=None,
            help="Official Uni-DMFM checkpoint variant.",
        )


__all__ = ["UniDMFMConfig", "UniDMFMFinetuneDefaults"]
