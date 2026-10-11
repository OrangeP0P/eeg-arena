"""Configuration and all public SleepFM Clinical checkpoint variants."""


class SleepFMClinicalConfig:
    MODEL_KEYS = ["classifier_dropout"]
    WEIGHTS_DIR = "./models/Weights/SleepFM"
    WEIGHT_VARIANTS = {
        "base-upstream": {
            "file": "model_base_best.pt",
            "description": "Official contrastive SetTransformer encoder",
        },
        "base-safetensors": {
            "file": "model.safetensors",
            "description": "Bit-identical Braindecode safetensors conversion of the base encoder",
        },
        "sleep-staging": {
            "file": "model_sleep_staging_best.pth",
            "description": "Official base encoder plus released sleep-staging transfer stack",
        },
        "diagnosis": {
            "file": "model_diagnosis_best.pth",
            "description": "Official base encoder plus released clinical diagnosis transfer stack",
        },
    }
    DEFAULT_VARIANT = "base-upstream"

    @staticmethod
    def get_model_keys():
        return SleepFMClinicalConfig.MODEL_KEYS.copy()

    @staticmethod
    def get_input_constraints():
        return {"required_input_format": "raw", "supports_pretrained": True}

    @staticmethod
    def add_args(parser, mode="all"):
        if mode not in {"all", "finetune"}:
            return
        group = parser.add_argument_group("SleepFM Clinical fine-tune options")
        group.add_argument("--classifier_dropout", type=float, default=0.1)
        group.add_argument(
            "--foundation_dir",
            type=str,
            default="./models/Weights/SleepFM",
        )
        group.add_argument(
            "--variant",
            choices=tuple(SleepFMClinicalConfig.WEIGHT_VARIANTS),
            default=None,
        )
