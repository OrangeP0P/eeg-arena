"""CodeBrain model config for the unified fine-tuning framework."""
from dataclasses import dataclass


@dataclass
class CodeBrainFinetuneDefaults:
    """CodeBrain finetune defaults for IDE auto-completion only."""

    d_model: int = 200
    n_layer: int = 8
    dropout: float = 0.1
    codebook_size_t: int = 4096
    codebook_size_f: int = 4096
    frozen: bool = False
    use_pretrained_weights: bool = True
    foundation_dir: str = "./models/Weights/CodeBrain/CodeBrain.pth"


class CodeBrainConfig:
    """CodeBrain configuration class."""

    MODEL_KEYS = ["d_model", "n_layer", "dropout", "codebook_size_t", "codebook_size_f"]

    @staticmethod
    def get_model_keys():
        return CodeBrainConfig.MODEL_KEYS.copy()

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
            CodeBrainConfig._add_finetune_args(parser)

    @staticmethod
    def _add_finetune_args(parser):
        group = parser.add_argument_group("CodeBrain Fine-tune options")
        group.add_argument(
            "--d_model",
            type=int,
            default=200,
            help="Backbone hidden dimension (default: 200)",
        )
        group.add_argument(
            "--n_layer",
            type=int,
            default=8,
            help="Number of residual layers in SSSM (default: 8)",
        )
        group.add_argument(
            "--dropout",
            type=float,
            default=0.1,
            help="Dropout used by SSSM and classifier head (default: 0.1)",
        )
        group.add_argument(
            "--codebook_size_t",
            type=int,
            default=4096,
            help="Temporal codebook size used by the pretrained backbone definition (default: 4096)",
        )
        group.add_argument(
            "--codebook_size_f",
            type=int,
            default=4096,
            help="Frequency codebook size used by the pretrained backbone definition (default: 4096)",
        )
        group.add_argument(
            "--foundation_dir",
            type=str,
            default="./models/Weights/CodeBrain/CodeBrain.pth",
            help="Path to CodeBrain pretrained weights file or directory",
        )
