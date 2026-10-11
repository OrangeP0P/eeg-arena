"""Configuration for the public SleepGPT release."""


class SleepGPTConfig:
    MODEL_KEYS = ["classifier_dropout", "sleepgpt_arch"]
    WEIGHTS_DIR = "./models/Weights/SleepGPT"
    WEIGHT_VARIANTS = {
        "large-pretrained": {
            "file": "ModelCheckpoint-epoch=79-val_acc=0.0000-val_score=4.2305.ckpt",
            "description": "Official 12-layer time-frequency SleepGPT checkpoint",
        },
    }
    DEFAULT_VARIANT = "large-pretrained"

    @staticmethod
    def get_model_keys():
        return SleepGPTConfig.MODEL_KEYS.copy()

    @staticmethod
    def get_input_constraints():
        return {"required_input_format": "raw", "supports_pretrained": True}

    @staticmethod
    def add_args(parser, mode="all"):
        if mode not in {"all", "finetune"}:
            return
        group = parser.add_argument_group("SleepGPT fine-tune options")
        group.add_argument("--classifier_dropout", type=float, default=0.1)
        group.add_argument(
            "--sleepgpt_arch",
            choices=("base", "large", "huge"),
            default="large",
            help=(
                "Official code architecture. Only large has a public pretrained "
                "checkpoint; base/huge remain available for architecture ablations."
            ),
        )
        group.add_argument(
            "--foundation_dir",
            type=str,
            default=(
                "./models/Weights/SleepGPT/"
                "ModelCheckpoint-epoch=79-val_acc=0.0000-val_score=4.2305.ckpt"
            ),
        )
        group.add_argument(
            "--variant",
            choices=tuple(SleepGPTConfig.WEIGHT_VARIANTS),
            default=None,
        )
