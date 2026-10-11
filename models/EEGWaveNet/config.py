"""EEGWaveNet config."""


class EEGWaveNetConfig:
    MODEL_KEYS = []

    @staticmethod
    def get_model_keys():
        return EEGWaveNetConfig.MODEL_KEYS.copy()

    @staticmethod
    def get_input_constraints():
        return {"required_input_format": "raw", "supports_pretrained": False}

    @staticmethod
    def add_args(parser, mode="all"):
        del parser, mode
