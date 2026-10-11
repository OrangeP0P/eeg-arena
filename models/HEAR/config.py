"""HEAR model configuration for the unified fine-tuning framework."""
import argparse
from dataclasses import dataclass


HEAR_WEIGHT_VARIANTS = {
    "base": {
        "file": "base/checkpoint.pth",
        "description": "HEAR base checkpoint (6 layers, 4 attention heads)",
        "depth": 6,
        "num_heads": 4,
        "layer_scale_init_value": 0.1,
    },
    "large": {
        "file": "large/checkpoint-49.pth",
        "description": "HEAR large checkpoint (12 layers, 8 attention heads)",
        "depth": 12,
        "num_heads": 8,
        "layer_scale_init_value": 1e-4,
    },
}


@dataclass
class HEARFinetuneDefaults:
    """HEAR fine-tune defaults for IDE auto-completion only."""

    variant: str = "base"
    embed_dim: int = 200
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    dropout: float = 0.5
    layer_scale_init_value: float | None = None
    init_scale: float = 0.001
    use_abs_pos_emb: bool = True
    use_rel_pos_bias: bool = True
    use_mean_pooling: bool = True
    hear_channel_mode: str = "heterogeneous"
    frozen: bool = False
    use_pretrained_weights: bool = True
    foundation_dir: str = "./models/Weights/HEAR/base/checkpoint.pth"


class HEARConfig:
    """HEAR configuration class."""

    MODEL_KEYS = [
        "embed_dim",
        "mlp_ratio",
        "qkv_bias",
        "drop_rate",
        "attn_drop_rate",
        "drop_path_rate",
        "dropout",
        "layer_scale_init_value",
        "init_scale",
        "use_abs_pos_emb",
        "use_rel_pos_bias",
        "use_mean_pooling",
        "hear_channel_mode",
    ]

    WEIGHT_VARIANTS = HEAR_WEIGHT_VARIANTS
    DEFAULT_VARIANT = "base"
    WEIGHTS_DIR = "./models/Weights/HEAR"

    @staticmethod
    def get_model_keys():
        return HEARConfig.MODEL_KEYS.copy()

    @staticmethod
    def get_input_constraints():
        return {
            "required_input_format": "patched",
            "required_patch_size": 200,
            "supports_pretrained": True,
        }

    @staticmethod
    def add_args(parser, mode="all"):
        if mode in ["all", "finetune"]:
            HEARConfig._add_finetune_args(parser)

    @staticmethod
    def _add_finetune_args(parser):
        group = parser.add_argument_group("HEAR Fine-tune options")

        group.add_argument(
            "--variant",
            type=str,
            default=None,
            choices=list(HEARConfig.WEIGHT_VARIANTS.keys()),
            help="HEAR pretrained-weight variant (base or large).",
        )
        group.add_argument(
            "--foundation_dir",
            type=str,
            default="./models/Weights/HEAR/base/checkpoint.pth",
            help="Path to a HEAR checkpoint file or checkpoint directory.",
        )
        group.add_argument("--embed_dim", type=int, default=200, help="Transformer embedding dimension.")
        group.add_argument("--mlp_ratio", type=float, default=4.0, help="Transformer MLP expansion ratio.")
        group.add_argument(
            "--qkv_bias",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Enable attention QKV bias (default: True).",
        )
        group.add_argument("--drop_rate", type=float, default=0.0, help="Backbone dropout rate.")
        group.add_argument("--attn_drop_rate", type=float, default=0.0, help="Attention dropout rate.")
        group.add_argument("--drop_path_rate", type=float, default=0.1, help="Drop-path rate.")
        group.add_argument("--dropout", type=float, default=0.5, help="Classifier dropout rate.")
        group.add_argument(
            "--layer_scale_init_value",
            type=float,
            default=None,
            help="Layer-scale init value. Defaults to the selected HEAR variant.",
        )
        group.add_argument("--init_scale", type=float, default=0.001, help="Classifier init scale.")
        group.add_argument(
            "--use_abs_pos_emb",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Use HEAR absolute electrode embeddings (default: True).",
        )
        group.add_argument(
            "--use_rel_pos_bias",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Use HEAR relative spatial bias (default: True).",
        )
        group.add_argument(
            "--use_mean_pooling",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Use mean-pooled patch tokens instead of CLS token (default: True).",
        )
        group.add_argument(
            "--hear_channel_mode",
            type=str,
            default="heterogeneous",
            choices=("heterogeneous", "standard_1020"),
            help=(
                "HEAR channel-coordinate mode. 'heterogeneous' uses dataloader "
                "runtime coordinates; 'standard_1020' uses HEAR's fixed 108-electrode table."
            ),
        )
