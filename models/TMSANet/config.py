"""TMSANet config."""
from dataclasses import dataclass


@dataclass
class TMSANetDefaults:
    embed_dim: int = 32
    pool_size: int = 50
    pool_stride: int = 15
    heads: int = 4
    depth: int = 2
    dropout: float = 0.5


class TMSANetConfig:
    MODEL_KEYS = ["embed_dim", "pool_size", "pool_stride", "heads", "depth", "dropout"]

    @staticmethod
    def get_model_keys():
        return TMSANetConfig.MODEL_KEYS.copy()

    @staticmethod
    def get_input_constraints():
        return {"required_input_format": "raw", "supports_pretrained": False}

    @staticmethod
    def add_args(parser, mode="all"):
        if mode in ["all", "finetune"]:
            group = parser.add_argument_group("TMSANet Fine-tune options")
            group.add_argument("--embed_dim", type=int, default=32, help="Embedding size after the CNN stem (default: 32)")
            group.add_argument("--pool_size", type=int, default=50, help="Temporal pooling size (default: 50)")
            group.add_argument("--pool_stride", type=int, default=15, help="Temporal pooling stride (default: 15)")
            group.add_argument("--heads", type=int, default=4, help="Number of transformer heads (default: 4)")
            group.add_argument("--depth", type=int, default=2, help="Number of transformer blocks (default: 2)")
            group.add_argument("--dropout", type=float, default=0.5, help="Dropout rate (default: 0.5)")
