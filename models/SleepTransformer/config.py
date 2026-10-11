"""SleepTransformer config."""
from dataclasses import dataclass


@dataclass
class SleepTransformerDefaults:
    frame_size: int = 128
    frame_step: int = 64
    frame_d_model: int = 128
    frame_heads: int = 4
    frame_depth: int = 2
    seq_d_model: int = 128
    seq_heads: int = 4
    seq_depth: int = 2
    epoch_seq_len: int = 4
    dropout: float = 0.3


class SleepTransformerConfig:
    MODEL_KEYS = [
        "frame_size",
        "frame_step",
        "frame_d_model",
        "frame_heads",
        "frame_depth",
        "seq_d_model",
        "seq_heads",
        "seq_depth",
        "epoch_seq_len",
        "dropout",
    ]

    @staticmethod
    def get_model_keys():
        return SleepTransformerConfig.MODEL_KEYS.copy()

    @staticmethod
    def get_input_constraints():
        return {"required_input_format": "raw", "supports_pretrained": False}

    @staticmethod
    def add_args(parser, mode="all"):
        if mode in ["all", "finetune"]:
            group = parser.add_argument_group("SleepTransformer Fine-tune options")
            group.add_argument("--frame_size", type=int, default=128, help="Frame size in samples (default: 128)")
            group.add_argument("--frame_step", type=int, default=64, help="Stride between adjacent frames (default: 64)")
            group.add_argument("--frame_d_model", type=int, default=128, help="Frame-level transformer width (default: 128)")
            group.add_argument("--frame_heads", type=int, default=4, help="Frame-level attention heads (default: 4)")
            group.add_argument("--frame_depth", type=int, default=2, help="Frame-level transformer depth (default: 2)")
            group.add_argument("--seq_d_model", type=int, default=128, help="Epoch-level transformer width (default: 128)")
            group.add_argument("--seq_heads", type=int, default=4, help="Epoch-level attention heads (default: 4)")
            group.add_argument("--seq_depth", type=int, default=2, help="Epoch-level transformer depth (default: 2)")
            group.add_argument("--epoch_seq_len", type=int, default=4, help="Number of coarse epoch tokens (default: 4)")
            group.add_argument("--dropout", type=float, default=0.3, help="Dropout rate (default: 0.3)")
