"""LUNA model configuration."""
from dataclasses import dataclass


@dataclass
class LUNAFinetuneDefaults:
    """LUNA fine-tune defaults for IDE auto-completion only."""

    classifier_dropout: float = 0.15
    foundation_dir: str = "./models/Weights/LUNA/LUNA_base.safetensors"


class LUNAConfig:
    """LUNA configuration class."""

    MODEL_KEYS = ["classifier_dropout"]

    WEIGHT_VARIANTS = {
        "base": {
            "file": "LUNA_base.safetensors",
            "description": "LUNA base checkpoint (8 blocks, 4 queries, 64 dim)",
            "patch_size": 40,
            "num_queries": 4,
            "embed_dim": 64,
            "depth": 8,
            "num_heads": 2,
            "mlp_ratio": 4.0,
        },
        "large": {
            "file": "LUNA_large.safetensors",
            "description": "LUNA large checkpoint (10 blocks, 6 queries, 96 dim)",
            "patch_size": 40,
            "num_queries": 6,
            "embed_dim": 96,
            "depth": 10,
            "num_heads": 2,
            "mlp_ratio": 4.0,
        },
        "huge": {
            "file": "LUNA_huge.safetensors",
            "description": "LUNA huge checkpoint (24 blocks, 8 queries, 128 dim)",
            "patch_size": 40,
            "num_queries": 8,
            "embed_dim": 128,
            "depth": 24,
            "num_heads": 2,
            "mlp_ratio": 4.0,
        },
    }
    DEFAULT_VARIANT = "base"
    WEIGHTS_DIR = "./models/Weights/LUNA"

    @staticmethod
    def get_model_keys():
        return LUNAConfig.MODEL_KEYS.copy()

    @staticmethod
    def get_input_constraints():
        return {
            "required_input_format": "raw",
            "supports_pretrained": True,
        }

    @staticmethod
    def add_args(parser, mode="all"):
        if mode in ["all", "finetune"]:
            LUNAConfig._add_finetune_args(parser)

    @staticmethod
    def _add_finetune_args(parser):
        group = parser.add_argument_group("LUNA Fine-tune options")

        group.add_argument(
            "--classifier_dropout",
            type=float,
            default=0.15,
            help="Dropout used in the LUNA classification head (default: 0.15).",
        )
        group.add_argument(
            "--foundation_dir",
            type=str,
            default="./models/Weights/LUNA/LUNA_base.safetensors",
            help="Path to a LUNA checkpoint file or checkpoint directory.",
        )
        group.add_argument(
            "--variant",
            type=str,
            default=None,
            choices=list(LUNAConfig.WEIGHT_VARIANTS.keys()),
            help="Pretrained weight variant: "
            + ", ".join(
                f"{name} ({info['description']})"
                for name, info in LUNAConfig.WEIGHT_VARIANTS.items()
            ),
        )
