"""IFNet config."""
from dataclasses import dataclass


@dataclass
class IFNetDefaults:
    out_planes: int = 64
    radix: int = 3
    kernel_size: int = 63
    ifnet_patch_size: int = 125
    dropout: float = 0.5


class IFNetConfig:
    MODEL_KEYS = ["out_planes", "radix", "kernel_size", "ifnet_patch_size", "dropout"]

    @staticmethod
    def get_model_keys():
        return IFNetConfig.MODEL_KEYS.copy()

    @staticmethod
    def get_input_constraints():
        return {"required_input_format": "raw", "supports_pretrained": False}

    @staticmethod
    def add_args(parser, mode="all"):
        if mode in ["all", "finetune"]:
            group = parser.add_argument_group("IFNet Fine-tune options")
            group.add_argument("--out_planes", type=int, default=64, help="Hidden channel size (default: 64)")
            group.add_argument("--radix", type=int, default=3, help="Number of interactive frequency branches (default: 3)")
            group.add_argument("--kernel_size", type=int, default=63, help="Largest temporal kernel size (default: 63)")
            group.add_argument(
                "--ifnet_patch_size",
                type=int,
                default=125,
                help="Temporal patch length used by the log-power pooling head (default: 125)",
            )
            group.add_argument("--dropout", type=float, default=0.5, help="Dropout rate (default: 0.5)")
