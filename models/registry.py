"""
Model registry implementation colocated with the models package.
"""
import importlib
import os

# Model registry
MODEL_REGISTRY = {}


def register_model(name, model_class, config_class=None, dataset_adapter_class=None,
                  pretrain_adapter_class=None):
    """
    Register a model.

    Args:
        name: model name (lowercase)
        model_class: model class
        config_class: model config class (optional)
        dataset_adapter_class: fine-tune dataset adapter class (optional)
        pretrain_adapter_class: pretrain dataset adapter class (optional)
    """
    MODEL_REGISTRY[name.lower()] = {
        'model_class': model_class,
        'config_class': config_class,
        'dataset_adapter_class': dataset_adapter_class,
        'pretrain_adapter_class': pretrain_adapter_class,
    }


def get_model(name):
    """Get model class by name."""
    return MODEL_REGISTRY[name.lower()]['model_class']


def get_model_config(name):
    """Get model config class by name."""
    config_class = MODEL_REGISTRY[name.lower()].get('config_class')
    if config_class is None:
        return DefaultConfig
    return config_class


def get_finetune_adapter(name):
    """Get dataset adapter class by name (for fine-tuning)."""
    adapter_class = MODEL_REGISTRY[name.lower()].get('dataset_adapter_class')
    if adapter_class is None:
        return None
    return adapter_class


def get_pretrain_adapter(name):
    """Get pretrain dataset adapter class by name."""
    adapter_class = MODEL_REGISTRY[name.lower()].get('pretrain_adapter_class')
    if adapter_class is None:
        return None
    return adapter_class


def list_models():
    """List all registered models."""
    return list(MODEL_REGISTRY.keys())


def has_model(name):
    """Check if model is registered."""
    return name.lower() in MODEL_REGISTRY


class DefaultConfig:
    """Default config class for models without special parameters."""

    @staticmethod
    def get_model_keys():
        """Get the list of params required for model construction."""
        return []

    @staticmethod
    def add_args(parser, mode='finetune'):
        """Add model-specific arguments (default: none)."""
        del parser, mode


def _autoload_models():
    """Auto-scan and load model packages from models/ directory."""
    models_dir = os.path.dirname(__file__)
    if not os.path.isdir(models_dir):
        return

    for item in os.listdir(models_dir):
        item_path = os.path.join(models_dir, item)
        init_file = os.path.join(item_path, "__init__.py")

        # Only import valid model packages under models/*
        if not os.path.isdir(item_path):
            continue
        if not os.path.exists(init_file):
            continue
        if item.startswith('_') or item.startswith('.'):
            continue

        try:
            importlib.import_module(f"models.{item}")
        except Exception as e:
            print(f"[models.registry] Failed to import models.{item}: {e}")


_autoload_models()


__all__ = [
    "MODEL_REGISTRY",
    "DefaultConfig",
    "get_finetune_adapter",
    "get_model",
    "get_model_config",
    "get_pretrain_adapter",
    "has_model",
    "list_models",
    "register_model",
]
