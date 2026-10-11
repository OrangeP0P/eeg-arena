"""LEAD model config."""
from dataclasses import dataclass


@dataclass
class LEADFinetuneDefaults:
    """LEAD defaults for IDE hints."""

    d_model: int = 128
    n_heads: int = 8
    e_layers: int = 12
    d_ff: int = 256
    dropout: float = 0.1
    activation: str = "gelu"
    patch_len: int = 50
    stride: int = 50
    augmentations: str = "flip,frequency,jitter,mask,channel,drop"
    sampling_rate: int = 200
    frozen: bool = False
    use_pretrained_weights: bool = True
    foundation_dir: str = "./models/Weights/LEAD/p-base"


class LEADConfig:
    """LEAD configuration class."""

    MODEL_KEYS = [
        "d_model",
        "n_heads",
        "e_layers",
        "d_ff",
        "dropout",
        "activation",
        "patch_len",
        "stride",
        "augmentations",
        "sampling_rate",
    ]

    @staticmethod
    def get_model_keys():
        return LEADConfig.MODEL_KEYS.copy()

    @staticmethod
    def get_input_constraints():
        return {
            "required_input_format": "raw",
            "supports_pretrained": True,
        }

    @staticmethod
    def add_args(parser, mode="all"):
        if mode in ["all", "finetune"]:
            LEADConfig._add_finetune_args(parser)

    @staticmethod
    def _add_finetune_args(parser):
        group = parser.add_argument_group("LEAD Fine-tune options")
        group.add_argument(
            "--d_model",
            type=int,
            default=128,
            help="Transformer embedding dimension for LEADv2 (default: 128).",
        )
        group.add_argument(
            "--n_heads",
            type=int,
            default=8,
            help="Number of attention heads in LEADv2 (default: 8).",
        )
        group.add_argument(
            "--e_layers",
            type=int,
            default=12,
            help="Number of LEADv2 encoder layers (default: 12).",
        )
        group.add_argument(
            "--d_ff",
            type=int,
            default=256,
            help="Feed-forward hidden dimension in LEADv2 encoder layers (default: 256).",
        )
        group.add_argument(
            "--dropout",
            type=float,
            default=0.1,
            help="Dropout used in LEADv2 backbone and classifier (default: 0.1).",
        )
        group.add_argument(
            "--activation",
            type=str,
            default="gelu",
            choices=["relu", "gelu"],
            help="Activation function used in LEADv2 encoder blocks (default: gelu).",
        )
        group.add_argument(
            "--patch_len",
            type=int,
            default=50,
            help="Patch length used by LEADv2's internal patch embedding (default: 50).",
        )
        group.add_argument(
            "--stride",
            type=int,
            default=50,
            help="Stride used by LEADv2's internal patch embedding (default: 50).",
        )
        group.add_argument(
            "--augmentations",
            type=str,
            default="flip,frequency,jitter,mask,channel,drop",
            help="Comma-separated LEADv2 augmentations applied during training.",
        )
        group.add_argument(
            "--sampling_rate",
            type=int,
            default=200,
            help="Sampling rate token fed into LEADv2. Current downstream datasets use 200 Hz.",
        )
        group.add_argument(
            "--foundation_dir",
            type=str,
            default="./models/Weights/LEAD/p-base",
            help="Path to a LEAD checkpoint file or checkpoint directory.",
        )
