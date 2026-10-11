"""DBConformer config."""
from dataclasses import dataclass


@dataclass
class DBConformerDefaults:
    emb_size: int = 40
    tem_depth: int = 4
    spa_depth: int = 4
    radix: int = 2
    kernel_size: int = 63
    temporal_patch_size: int = 125
    spa_dim: int = 32
    dropout: float = 0.5


class DBConformerConfig:
    MODEL_KEYS = [
        "emb_size",
        "tem_depth",
        "spa_depth",
        "radix",
        "kernel_size",
        "temporal_patch_size",
        "spa_dim",
        "dropout",
    ]

    @staticmethod
    def get_model_keys():
        return DBConformerConfig.MODEL_KEYS.copy()

    @staticmethod
    def get_input_constraints():
        return {"required_input_format": "raw", "supports_pretrained": False}

    @staticmethod
    def add_args(parser, mode="all"):
        if mode in ["all", "finetune"]:
            group = parser.add_argument_group("DBConformer Fine-tune options")
            group.add_argument("--emb_size", type=int, default=40, help="Branch embedding size (default: 40)")
            group.add_argument("--tem_depth", type=int, default=4, help="Temporal transformer depth (default: 4)")
            group.add_argument("--spa_depth", type=int, default=4, help="Spatial transformer depth (default: 4)")
            group.add_argument("--radix", type=int, default=2, help="Temporal multi-branch radix (default: 2)")
            group.add_argument("--kernel_size", type=int, default=63, help="Largest temporal kernel size (default: 63)")
            group.add_argument(
                "--temporal_patch_size",
                type=int,
                default=125,
                help="Temporal pooling size for tokenization (default: 125)",
            )
            group.add_argument("--spa_dim", type=int, default=32, help="Spatial branch hidden size (default: 32)")
            group.add_argument("--dropout", type=float, default=0.5, help="Dropout rate (default: 0.5)")
