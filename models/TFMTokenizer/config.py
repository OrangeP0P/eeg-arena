"""TFM-Tokenizer/TFM-Encoder configuration and full official weight catalogue."""
from dataclasses import dataclass


_MULTI_TOKENIZER = "huggingface/pretrained/tfm_tokenizer_last.pth"


@dataclass
class TFMTokenizerFinetuneDefaults:
    foundation_dir: str = "./models/Weights/TFMTokenizer"
    tfm_tokenizer_dir: str | None = None


class TFMTokenizerConfig:
    MODEL_KEYS = ["tfm_tokenizer_dir"]
    WEIGHTS_DIR = "./models/Weights/TFMTokenizer"
    _MULTI_TOKENIZER = _MULTI_TOKENIZER
    WEIGHT_VARIANTS = {
        "multi-pretrained": {
            "file": "huggingface/pretrained/tfm_encoder_mtp_last.pth",
            "tokenizer_file": _MULTI_TOKENIZER,
            "description": "MTP-pretrained TFM-Encoder with the multi-dataset tokenizer",
        },
        "multi-chbmit": {
            "file": "github/multiple_dataset_settings/Finetuned_CHBMIT/tfm_encoder_best_model.pth",
            "tokenizer_file": _MULTI_TOKENIZER,
            "description": "Multi-dataset tokenizer, CHB-MIT fine-tuned encoder",
        },
        "multi-tuev": {
            "file": "github/multiple_dataset_settings/Finetuned_TUEV/tfm_encoder_best_model.pth",
            "tokenizer_file": _MULTI_TOKENIZER,
            "description": "Multi-dataset tokenizer, TUEV fine-tuned encoder",
        },
        **{
            f"multi-tuab-seed-{seed}": {
                "file": f"huggingface/finetuned/TUAB/seed_{seed}/best_model.pth",
                "tokenizer_file": _MULTI_TOKENIZER,
                "description": f"Multi-dataset tokenizer, TUAB fine-tuned encoder seed {seed}",
            }
            for seed in range(1, 6)
        },
        **{
            f"multi-tuev-seed-{seed}": {
                "file": f"huggingface/finetuned/TUEV/seed_{seed}/best_model.pth",
                "tokenizer_file": _MULTI_TOKENIZER,
                "description": f"Multi-dataset tokenizer, TUEV fine-tuned encoder seed {seed}",
            }
            for seed in range(1, 6)
        },
        "single-chbmit": {
            "file": "github/single_dataset_settings/CHBMIT_tfm_tokenizer_2x2x8/tfm_encoder_best_model.pth",
            "tokenizer_file": "github/single_dataset_settings/CHBMIT_tfm_tokenizer_2x2x8/tfm_tokenizer_last.pth",
            "description": "CHB-MIT-only tokenizer and fine-tuned encoder",
        },
        "single-tuab": {
            "file": "github/single_dataset_settings/TUAB_tfm_tokenizer_2x2x8/tfm_encoder_best_model.pth",
            "tokenizer_file": "github/single_dataset_settings/TUAB_tfm_tokenizer_2x2x8/tfm_tokenizer_last.pth",
            "description": "TUAB-only tokenizer and fine-tuned encoder",
        },
        "single-tuev": {
            "file": "github/single_dataset_settings/TUEV_tfm_tokenizer_2x2x8/tfm_encoder_best_model.pth",
            "tokenizer_file": "github/single_dataset_settings/TUEV_tfm_tokenizer_2x2x8/tfm_tokenizer_last.pth",
            "description": "TUEV-only tokenizer and fine-tuned encoder",
        },
    }
    DEFAULT_VARIANT = "multi-pretrained"
    DEFAULT_WEIGHT_PATH = "./models/Weights/TFMTokenizer"

    @staticmethod
    def get_model_keys():
        return TFMTokenizerConfig.MODEL_KEYS.copy()

    @staticmethod
    def get_input_constraints():
        return {"required_input_format": "raw", "supports_pretrained": True}

    @staticmethod
    def add_args(parser, mode="all"):
        if mode not in {"all", "finetune"}:
            return
        group = parser.add_argument_group("TFM-Tokenizer/Encoder fine-tune options")
        group.add_argument(
            "--tfm_tokenizer_dir",
            default=None,
            help="Optional tokenizer checkpoint override; defaults to the selected paired release.",
        )
        group.add_argument(
            "--foundation_dir",
            default=TFMTokenizerConfig.DEFAULT_WEIGHT_PATH,
            help="TFM-Encoder checkpoint file or TFMTokenizer weights directory.",
        )
        group.add_argument(
            "--variant",
            choices=list(TFMTokenizerConfig.WEIGHT_VARIANTS),
            default=None,
            help="One of the official multi-dataset, seed, or single-dataset checkpoint pairs.",
        )


__all__ = ["TFMTokenizerConfig", "TFMTokenizerFinetuneDefaults"]
