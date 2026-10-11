# Downstream task (Fine-tune) data loader
from .finetune import (
    get_dataset,
    load_data,
    EpochDataset,
    SubjectSampler,
    ClusterBatchSampler,
    FinetuneReshapeDataset,
)

__all__ = [
    # Fine-tune
    "get_dataset",
    "load_data",
    "EpochDataset",
    "SubjectSampler",
    "ClusterBatchSampler",
    "FinetuneReshapeDataset",
]


def __getattr__(name):
    """Lazy import pretrain module"""
    if name in ['PretrainDataLoader', 'build_pretrain_datasets', 'generate_datasets_train',
                'generate_datasets', 'process_channel_location', 'SingleShockDataset', 'ShockDataset']:
        from . import pretrain
        return getattr(pretrain, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
