"""ATCNet config."""
from dataclasses import dataclass


@dataclass
class ATCNetDefaults:
    """ATCNet defaults for IDE hints."""

    n_windows: int = 5
    attention: str = "mha"
    attention_heads: int = 2
    attention_key_dim: int = 8
    attention_dropout: float = 0.5
    attention_residual_dropout: float = 0.3
    attention_se_ratio: int = 8
    eegn_F1: int = 16
    eegn_D: int = 2
    eegn_kernel_size: int = 64
    eegn_pool_size: int = 7
    eegn_dropout: float = 0.3
    tcn_depth: int = 2
    tcn_kernel_size: int = 4
    tcn_filters: int = 32
    tcn_dropout: float = 0.3
    tcn_activation: str = "elu"
    fuse: str = "average"


class ATCNetConfig:
    """ATCNet model configuration."""

    MODEL_KEYS = [
        "n_windows",
        "attention",
        "attention_heads",
        "attention_key_dim",
        "attention_dropout",
        "attention_residual_dropout",
        "attention_se_ratio",
        "eegn_F1",
        "eegn_D",
        "eegn_kernel_size",
        "eegn_pool_size",
        "eegn_dropout",
        "tcn_depth",
        "tcn_kernel_size",
        "tcn_filters",
        "tcn_dropout",
        "tcn_activation",
        "fuse",
    ]

    @staticmethod
    def get_model_keys():
        return ATCNetConfig.MODEL_KEYS.copy()

    @staticmethod
    def get_input_constraints():
        return {
            "required_input_format": "raw",
            "supports_pretrained": False,
        }

    @staticmethod
    def add_args(parser, mode="all"):
        if mode in ["all", "finetune"]:
            ATCNetConfig._add_finetune_args(parser)

    @staticmethod
    def _add_finetune_args(parser):
        group = parser.add_argument_group("ATCNet Fine-tune options")
        group.add_argument(
            "--n_windows",
            type=int,
            default=5,
            help="Number of parallel sliding-window branches (default: 5).",
        )
        group.add_argument(
            "--attention",
            type=str,
            default="mha",
            choices=["none", "mha", "mhla", "se", "cbam"],
            help="Attention block used before each TCN branch (default: mha).",
        )
        group.add_argument(
            "--attention_heads",
            type=int,
            default=2,
            help="Number of attention heads for mha/mhla (default: 2).",
        )
        group.add_argument(
            "--attention_key_dim",
            type=int,
            default=8,
            help="Per-head key dimension for mha/mhla (default: 8).",
        )
        group.add_argument(
            "--attention_dropout",
            type=float,
            default=0.5,
            help="Attention probability dropout for mha/mhla (default: 0.5).",
        )
        group.add_argument(
            "--attention_residual_dropout",
            type=float,
            default=0.3,
            help="Dropout applied after attention output projection (default: 0.3).",
        )
        group.add_argument(
            "--attention_se_ratio",
            type=int,
            default=8,
            help="Reduction ratio used by se/cbam attention (default: 8).",
        )
        group.add_argument(
            "--eegn_F1",
            type=int,
            default=16,
            help="Number of temporal filters in the convolutional frontend (default: 16).",
        )
        group.add_argument(
            "--eegn_D",
            type=int,
            default=2,
            help="Depth multiplier for the spatial convolution block (default: 2).",
        )
        group.add_argument(
            "--eegn_kernel_size",
            type=int,
            default=64,
            help="Temporal kernel size in the convolutional frontend (default: 64).",
        )
        group.add_argument(
            "--eegn_pool_size",
            type=int,
            default=7,
            help="Second temporal pooling size in the convolutional frontend (default: 7).",
        )
        group.add_argument(
            "--eegn_dropout",
            type=float,
            default=0.3,
            help="Dropout rate in the convolutional frontend (default: 0.3).",
        )
        group.add_argument(
            "--tcn_depth",
            type=int,
            default=2,
            help="Number of residual TCN blocks per branch (default: 2).",
        )
        group.add_argument(
            "--tcn_kernel_size",
            type=int,
            default=4,
            help="Kernel size of the branch TCN blocks (default: 4).",
        )
        group.add_argument(
            "--tcn_filters",
            type=int,
            default=32,
            help="Hidden dimension of the branch TCN blocks (default: 32).",
        )
        group.add_argument(
            "--tcn_dropout",
            type=float,
            default=0.3,
            help="Dropout rate inside the branch TCN blocks (default: 0.3).",
        )
        group.add_argument(
            "--tcn_activation",
            type=str,
            default="elu",
            choices=["elu", "relu", "gelu", "silu"],
            help="Activation used in the branch TCN blocks (default: elu).",
        )
        group.add_argument(
            "--fuse",
            type=str,
            default="average",
            choices=["average", "concat"],
            help="Fuse window branches by averaging logits or concatenating features (default: average).",
        )
