"""mdJPT model configuration."""
import argparse
from dataclasses import dataclass


@dataclass
class MdJPTFinetuneDefaults:
    """mdJPT fine-tune defaults for IDE auto-completion only."""

    patch_len: int = 32
    patch_stride: int = 6
    d_model: int = 32
    n_heads: int = 8
    n_layers: int = 3
    d_ff: int = 256
    transformer_dropout: float = 0.0
    attn_dropout: float = 0.0

    target_channels: int = 64
    n_ms_filters: int = 4
    ms_filter_time_len: int = 3
    dilation_array: tuple[int, int, int, int] = (1, 3, 6, 12)
    seg_att: int = 15
    cnn_multi_fact: int = 1
    has_att: bool = True
    global_att: bool = False

    classifier_hidden_dim: int = 0
    feature_pooling: str = "mean"
    dropout: float = 0.1

    frozen: bool = False
    use_pretrained_weights: bool = True
    foundation_dir: str = "./models/Weights/mdJPT/mdjpt_pretrained.pth"


class MdJPTConfig:
    """mdJPT configuration class."""

    MODEL_KEYS = [
        "patch_len",
        "patch_stride",
        "d_model",
        "n_heads",
        "n_layers",
        "d_ff",
        "transformer_dropout",
        "attn_dropout",
        "target_channels",
        "n_ms_filters",
        "ms_filter_time_len",
        "dilation_array",
        "seg_att",
        "cnn_multi_fact",
        "has_att",
        "global_att",
        "classifier_hidden_dim",
        "feature_pooling",
        "dropout",
    ]

    @staticmethod
    def get_model_keys():
        return MdJPTConfig.MODEL_KEYS.copy()

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
            MdJPTConfig._add_finetune_args(parser)

    @staticmethod
    def _add_finetune_args(parser):
        group = parser.add_argument_group("mdJPT Fine-tune options")

        group.add_argument("--patch_len", type=int, default=32, help="Patch length for PatchTST encoder.")
        group.add_argument("--patch_stride", type=int, default=6, help="Patch stride for PatchTST encoder.")
        group.add_argument("--d_model", type=int, default=32, help="Transformer hidden size.")
        group.add_argument("--n_heads", type=int, default=8, help="Transformer attention heads.")
        group.add_argument("--n_layers", type=int, default=3, help="Number of PatchTST encoder layers.")
        group.add_argument("--d_ff", type=int, default=256, help="Transformer FFN hidden size.")
        group.add_argument(
            "--transformer_dropout",
            type=float,
            default=0.0,
            help="Dropout used by the PatchTST encoder.",
        )
        group.add_argument(
            "--attn_dropout",
            type=float,
            default=0.0,
            help="Attention dropout used by the PatchTST encoder.",
        )

        group.add_argument(
            "--target_channels",
            type=int,
            default=64,
            help="Channel count after mdJPT channel projection.",
        )
        group.add_argument("--n_ms_filters", type=int, default=4, help="Number of multi-scale filters.")
        group.add_argument(
            "--ms_filter_time_len",
            type=int,
            default=3,
            help="Temporal kernel size for multi-scale convolutions.",
        )
        group.add_argument(
            "--dilation_array",
            type=int,
            nargs="+",
            default=[1, 3, 6, 12],
            help="Four dilation values for mdJPT multi-scale convolutions.",
        )
        group.add_argument("--seg_att", type=int, default=15, help="Temporal segment size in the attention block.")
        group.add_argument(
            "--cnn_multi_fact",
            type=int,
            default=1,
            help="Expansion factor used by mdJPT projection conv layers.",
        )
        group.add_argument(
            "--has_att",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Enable mdJPT channel-temporal attention block.",
        )
        group.add_argument(
            "--global_att",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Use global attention pooling instead of local pooling in mdJPT attention block.",
        )

        group.add_argument(
            "--classifier_hidden_dim",
            type=int,
            default=0,
            help="Classifier hidden dim; <=0 uses mdJPT feature dim.",
        )
        group.add_argument(
            "--feature_pooling",
            type=str,
            choices=["mean", "max"],
            default="mean",
            help="Pooling mode for mdJPT feature maps.",
        )
        group.add_argument("--dropout", type=float, default=0.1, help="Classifier dropout.")

        group.add_argument(
            "--foundation_dir",
            type=str,
            default="./models/Weights/mdJPT/mdjpt_pretrained.pth",
            help="Path to mdJPT pretrained checkpoint file or directory.",
        )
