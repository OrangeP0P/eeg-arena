"""
EEGMamba model config for the unified fine-tuning framework.
"""
from dataclasses import dataclass


@dataclass
class EEGMambaFinetuneDefaults:
    """
    EEGMamba finetune defaults for IDE auto-completion only.
    """

    dropout: float = 0.1
    classifier: str = "all_patch_reps"
    frozen: bool = False
    use_pretrained_weights: bool = True
    foundation_dir: str = "./models/Weights/EEGMamba/pretrained_EEGMamba.pth"


class EEGMambaConfig:
    """EEGMamba configuration class."""

    MODEL_KEYS = ["dropout", "classifier"]

    @staticmethod
    def get_model_keys():
        """Get the list of parameters required for model construction."""
        return EEGMambaConfig.MODEL_KEYS.copy()

    @staticmethod
    def get_input_constraints():
        """
        EEGMamba consumes patched EEG inputs with a fixed patch size of 200.
        """
        return {
            "required_input_format": "patched",
            "required_patch_size": 200,
            "supports_pretrained": True,
        }

    @staticmethod
    def add_args(parser, mode="all"):
        """Add EEGMamba-specific arguments to argparse."""
        if mode in ["all", "finetune"]:
            EEGMambaConfig._add_finetune_args(parser)

    @staticmethod
    def _add_finetune_args(parser):
        group = parser.add_argument_group("EEGMamba Fine-tune options")
        group.add_argument(
            "--dropout",
            type=float,
            default=0.1,
            help="Dropout rate used by the finetune classifier head (default: 0.1)",
        )
        group.add_argument(
            "--classifier",
            type=str,
            default="all_patch_reps",
            choices=[
                "avgpooling_patch_reps",
                "all_patch_reps_onelayer",
                "all_patch_reps_twolayer",
                "all_patch_reps",
            ],
            help="Classifier head type (default: all_patch_reps)",
        )
        group.add_argument(
            "--foundation_dir",
            type=str,
            default="./models/Weights/EEGMamba/pretrained_EEGMamba.pth",
            help="Path to EEGMamba pretrained weights",
        )
