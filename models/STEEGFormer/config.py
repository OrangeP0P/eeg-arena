"""STEEGFormer configuration."""
from dataclasses import dataclass


@dataclass
class STEEGFormerFinetuneDefaults:
    """STEEGFormer fine-tune defaults for IDE auto-completion only."""

    classifier_dropout: float = 0.1
    foundation_dir: str = "./models/Weights/STEEGFormer"


class STEEGFormerConfig:
    """STEEGFormer configuration class."""

    MODEL_KEYS = ["classifier_dropout"]

    WEIGHT_VARIANTS = {
        "small": {
            "file": "small/checkpoint-300.pth",
            "description": "ST-EEGFormer small checkpoint (8 blocks, 512 dim)",
            "embed_dim": 512,
            "depth": 8,
            "num_heads": 8,
            "num_channel_embeddings": 145,
        },
        "base": {
            "file": "base/checkpoint-288.pth",
            "description": "ST-EEGFormer base checkpoint (12 blocks, 768 dim)",
            "embed_dim": 768,
            "depth": 12,
            "num_heads": 12,
            "num_channel_embeddings": 145,
        },
        "large": {
            "file": "large/large_weights_only_196.pth",
            "description": "ST-EEGFormer large checkpoint (24 blocks, 1024 dim)",
            "embed_dim": 1024,
            "depth": 24,
            "num_heads": 16,
            "num_channel_embeddings": 145,
        },
        "largev2": {
            "file": "large_v2/large_weights_only_210.pth",
            "description": "ST-EEGFormer large-v2 checkpoint with expanded channel embeddings",
            "embed_dim": 1024,
            "depth": 24,
            "num_heads": 16,
            "num_channel_embeddings": 256,
        },
    }
    DEFAULT_VARIANT = "base"
    WEIGHTS_DIR = "./models/Weights/STEEGFormer"

    @staticmethod
    def get_model_keys():
        return STEEGFormerConfig.MODEL_KEYS.copy()

    @staticmethod
    def get_input_constraints():
        return {
            "required_input_format": "raw",
            "supports_pretrained": True,
        }

    @staticmethod
    def add_args(parser, mode="all"):
        if mode in ["all", "finetune"]:
            STEEGFormerConfig._add_finetune_args(parser)

    @staticmethod
    def _add_finetune_args(parser):
        group = parser.add_argument_group("STEEGFormer Fine-tune options")

        group.add_argument(
            "--classifier_dropout",
            type=float,
            default=0.1,
            help="Dropout used before the ST-EEGFormer classification head (default: 0.1).",
        )
        group.add_argument(
            "--foundation_dir",
            type=str,
            default="./models/Weights/STEEGFormer",
            help="Path to a STEEGFormer checkpoint file or checkpoint directory.",
        )
        group.add_argument(
            "--variant",
            type=str,
            default=None,
            choices=list(STEEGFormerConfig.WEIGHT_VARIANTS.keys()),
            help="Pretrained weight variant: "
            + ", ".join(
                f"{name} ({info['description']})"
                for name, info in STEEGFormerConfig.WEIGHT_VARIANTS.items()
            ),
        )
