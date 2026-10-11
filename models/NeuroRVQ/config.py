"""NeuroRVQ model configuration."""
import argparse
from dataclasses import dataclass


@dataclass
class NeuroRVQFinetuneDefaults:
    """NeuroRVQ fine-tune defaults for IDE auto-completion only."""

    n_patches: int = 256
    out_chans: int = 8
    embed_dim: int = 200
    depth: int = 12
    num_heads: int = 10
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    init_values: float = 1e-5
    init_scale: float = 0.001
    classifier_hidden_dim: int = 0
    dropout: float = 0.1
    foundation_dir: str = "./models/Weights/NeuroRVQ/NeuroRVQ_EEG_foundation_model_v1.pt"


class NeuroRVQConfig:
    """NeuroRVQ configuration class."""

    MODEL_KEYS = [
        "n_patches",
        "out_chans",
        "embed_dim",
        "depth",
        "num_heads",
        "mlp_ratio",
        "qkv_bias",
        "drop_rate",
        "attn_drop_rate",
        "drop_path_rate",
        "init_values",
        "init_scale",
        "classifier_hidden_dim",
        "dropout",
    ]

    @staticmethod
    def get_model_keys():
        return NeuroRVQConfig.MODEL_KEYS.copy()

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
            NeuroRVQConfig._add_finetune_args(parser)

    @staticmethod
    def _add_finetune_args(parser):
        group = parser.add_argument_group("NeuroRVQ Fine-tune options")

        group.add_argument(
            "--n_patches",
            type=int,
            default=256,
            help="Maximum temporal embedding slots from pretrained NeuroRVQ (default: 256).",
        )
        group.add_argument("--out_chans", type=int, default=8, help="Temporal encoder output channels (default: 8).")
        group.add_argument("--embed_dim", type=int, default=200, help="Transformer embedding dimension (default: 200).")
        group.add_argument("--depth", type=int, default=12, help="Transformer depth (default: 12).")
        group.add_argument("--num_heads", type=int, default=10, help="Attention heads (default: 10).")
        group.add_argument("--mlp_ratio", type=float, default=4.0, help="MLP expansion ratio (default: 4.0).")
        group.add_argument(
            "--qkv_bias",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Enable attention QKV bias (default: True).",
        )
        group.add_argument("--drop_rate", type=float, default=0.0, help="Backbone dropout (default: 0.0).")
        group.add_argument("--attn_drop_rate", type=float, default=0.0, help="Attention dropout (default: 0.0).")
        group.add_argument("--drop_path_rate", type=float, default=0.0, help="Drop-path rate (default: 0.0).")
        group.add_argument("--init_values", type=float, default=1e-5, help="Layer-scale init value (default: 1e-5).")
        group.add_argument("--init_scale", type=float, default=0.001, help="Head init scale (default: 0.001).")
        group.add_argument(
            "--classifier_hidden_dim",
            type=int,
            default=0,
            help="Classifier hidden dim; <=0 uses single linear head.",
        )
        group.add_argument("--dropout", type=float, default=0.1, help="Classifier dropout (default: 0.1).")
        group.add_argument(
            "--foundation_dir",
            type=str,
            default="./models/Weights/NeuroRVQ/NeuroRVQ_EEG_foundation_model_v1.pt",
            help="Path to NeuroRVQ pretrained foundation checkpoint file or directory.",
        )
