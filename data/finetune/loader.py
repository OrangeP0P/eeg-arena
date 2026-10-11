"""
Finetune DataLoader - Unified fine-tuning data loader
Encapsulates data reading, patch conversion, DataLoader assembly
"""
from typing import Dict, Optional, Any

from torch.utils.data import DataLoader

from .core import get_dataset
from .sampling import validate_data_ratio
from .transforms import FinetuneReshapeDataset
from data.channels import get_model_channel_space


def _parse_param_scalar(value):
    if isinstance(value, (bool, int, float)) or value is None:
        return value

    text = str(value).strip()
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


def _parse_cli_kv_items(raw_items, option_name):
    if raw_items is None:
        return None

    if isinstance(raw_items, dict):
        return {
            str(key): _parse_param_scalar(value)
            for key, value in raw_items.items()
        }

    parsed = {}
    example = "tmin=-0.5 tmax=4.0" if option_name == "--epoch_param" else "resample=200 lower_bound=0.1"
    for kv in raw_items:
        if '=' not in kv:
            raise ValueError(
                f"{option_name} '{kv}' is not in key=value format. "
                f"Example: {option_name} {example}"
            )
        key, value = kv.split('=', 1)
        parsed[key] = _parse_param_scalar(value)
    return parsed

class FinetuneDataLoader:
    """
    Fine-tuning data loader - encapsulates the complete data processing pipeline

    Usage:
        loader = FinetuneDataLoader(args)
        data_loader = loader.get_data_loader()

    Auto-update args (if provided):
        - args.num_channels: number of EEG channels
        - args.patch_num: number of patches
        - args.num_of_classes: number of classification classes
    """

    SUPPORTED_FORMATS = ['patched', 'raw']

    def __init__(self, args: Optional[Any] = None, **legacy_kwargs):
        """
        Args:
            args: unified argparse-style object. Expected fields:
                - dataset, task, cache_dir
                - batch_size, fold, split_mode
                - patch_size, input_format, num_workers, channel_system
            legacy_kwargs: backward-compatible explicit arguments for old callsites.
        """
        if args is None and not legacy_kwargs:
            raise ValueError("FinetuneDataLoader expects an 'args' object or legacy keyword arguments.")

        self.args = args
        self.dataset = getattr(args, 'dataset', legacy_kwargs.get('dataset'))
        self.task = getattr(args, 'task', legacy_kwargs.get('task'))
        self.cache_dir = getattr(args, 'cache_dir', legacy_kwargs.get('cache_dir'))
        self.batch_size = getattr(args, 'batch_size', legacy_kwargs.get('batch_size', 256))
        self.fold = getattr(args, 'fold', legacy_kwargs.get('fold'))
        self.split_mode = getattr(args, 'split_mode', legacy_kwargs.get('split_mode', 'cs'))
        self.held_out_subject = getattr(args, 'held_out_subject', legacy_kwargs.get('held_out_subject'))
        self.patch_size = getattr(args, 'patch_size', legacy_kwargs.get('patch_size', 200))
        self.input_format = getattr(args, 'input_format', legacy_kwargs.get('input_format', 'patched'))
        self.num_workers = getattr(args, 'num_workers', legacy_kwargs.get('num_workers', 4))
        self.channel_system = getattr(args, 'channel_system', legacy_kwargs.get('channel_system', '10-20'))
        self.channel_variant = getattr(args, 'channel_variant', legacy_kwargs.get('channel_variant', None))
        self.channel_count_target = getattr(args, 'channel_count_target', legacy_kwargs.get('channel_count_target', None))
        self.channel_sparsity_config = getattr(args, 'channel_sparsity_config', legacy_kwargs.get('channel_sparsity_config', None))
        self.model_name = getattr(args, 'model', legacy_kwargs.get('model_name', None))
        self.variant = getattr(args, 'variant', legacy_kwargs.get('variant', None))
        self.seed = getattr(args, 'seed', legacy_kwargs.get('seed', 0))
        self.data_ratio = validate_data_ratio(getattr(args, 'data_ratio', legacy_kwargs.get('data_ratio', 1.0)))

        # Default fold is automatically determined based on split_mode
        if self.fold is None:
            self.fold = 1

        if self.dataset is None:
            raise ValueError("Missing required field: args.dataset")
        if self.task is None:
            raise ValueError("Missing required field: args.task")
        if self.cache_dir is None:
            raise ValueError("Missing required field: args.cache_dir")

        # Validate input_format
        if self.input_format not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported input_format: {self.input_format}. "
                f"Supported: {self.SUPPORTED_FORMATS}"
            )

        # Dataset properties (filled after initialization)
        self.num_classes: int = 0
        self.num_channels: int = 0
        self.patch_num: int = 0
        self.sample_counts: Dict[str, int] = {}

    def _make_build_params(self):
        """Construct HDF5BuildParams from args."""
        from data.raw_to_hdf5 import HDF5BuildParams

        args = self.args
        epoch_params = _parse_cli_kv_items(getattr(args, 'epoch_param', None), "--epoch_param")
        pipeline_params = _parse_cli_kv_items(getattr(args, 'pipeline_param', None), "--pipeline_param")

        return HDF5BuildParams(
            epoch_params=epoch_params,
            pipeline=getattr(args, 'pipeline', None),
            pipeline_params=pipeline_params,
            rebuild_hdf5=getattr(args, 'rebuild_hdf5', False),
            num_workers=getattr(args, 'num_workers', 16),
            exp_id=getattr(args, 'exp_id', None),
            run_id=getattr(args, 'run_id', None),
            usage_report={},
        )

    def get_data_loader(self) -> Dict[str, Any]:
        """
        Get complete data loader

        Returns:
            dict: {
                'train': DataLoader,
                'val': DataLoader,
                'test': DataLoader,
                'num_classes': int,
                'num_channels': int,
                'patch_num': int,
                'ch_positions': dict,  # {channel_name: [x, y, z]}
            }
        """
        # Resolve model-specific channel space
        model_channel_space = (
            get_model_channel_space(self.model_name, variant=self.variant)
            if self.model_name
            else None
        )

        # Build HDF5BuildParams
        build_params = self._make_build_params()

        # 1. Get raw data (C, T) tensor dataset
        train_ds_raw, val_ds_raw, test_ds_raw, ch_names, ch_positions, num_classes, num_channels, model_channel_indices, split_subjects, self.sample_counts, resolved_channel_count_target = get_dataset(
            self.dataset, self.task, self.fold,
            split_mode=self.split_mode,
            channel_system=self.channel_system,
            channel_variant=self.channel_variant,
            channel_count_target=self.channel_count_target,
            channel_sparsity_config=self.channel_sparsity_config,
            patch_size=self.patch_size,
            input_format=self.input_format,
            model_channel_space=model_channel_space,
            build_params=build_params,
            cache_dir=self.cache_dir,
            held_out_subject=self.held_out_subject,
            data_ratio=self.data_ratio,
            sampling_seed=self.seed,
        )

        print('ch_names', ch_names)
        cache_usage = dict(build_params.usage_report or {})
        if self.args is not None:
            self.args.cache_usage = cache_usage
        # Save metadata
        self.num_classes = num_classes
        self.num_channels = num_channels

        # 2. Convert to patch format or keep raw
        if self.input_format == 'patched':
            train_set = FinetuneReshapeDataset(train_ds_raw, patch_size=self.patch_size)
            val_set = FinetuneReshapeDataset(val_ds_raw, patch_size=self.patch_size)
            test_set = FinetuneReshapeDataset(test_ds_raw, patch_size=self.patch_size)

            # Get actual patch_num
            sample_x, _ = train_set[0]
            self.patch_num = sample_x.shape[1]  # patch_num
            time_len = sample_x.shape[1] * sample_x.shape[2]

            collate_fn = self._patch_collate
        else:  # raw
            train_set = train_ds_raw
            val_set = val_ds_raw
            test_set = test_ds_raw
            sample_x, _ = train_set[0]
            time_len = sample_x.shape[-1]
            self.patch_num = max(1, (time_len + self.patch_size - 1) // self.patch_size)

            collate_fn = self._raw_collate

        # Ensure no HDF5 handle opened during metadata probing is inherited by workers.
        self._close_dataset_handles(train_set, val_set, test_set)

        common_loader_kwargs = {
            'num_workers': self.num_workers,
            'collate_fn': collate_fn,
        }
        if self.num_workers > 0:
            common_loader_kwargs['persistent_workers'] = True

        train_drop_last = True
        if len(train_set) < self.batch_size:
            train_drop_last = False
            print(
                f"  Warning: Train samples ({len(train_set)}) < batch_size ({self.batch_size}); "
                "disabling drop_last for the train DataLoader."
            )

        # 3. Wrap as DataLoader
        data_loader = {
            'train': DataLoader(
                train_set,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=train_drop_last,
                **common_loader_kwargs,
            ),
            'val': DataLoader(
                val_set,
                batch_size=self.batch_size,
                shuffle=False,
                **common_loader_kwargs,
            ),
            'test': DataLoader(
                test_set,
                batch_size=self.batch_size,
                shuffle=False,
                **common_loader_kwargs,
            ),
            'num_classes': num_classes,
            'num_channels': num_channels,
            'patch_num': self.patch_num,
            'samples': time_len,
            'ch_positions': ch_positions,
            'ch_names': ch_names,
            'model_channel_indices': model_channel_indices,
            'channel_variant': self.channel_variant,
            'channel_count_target': resolved_channel_count_target,
            'channel_count_loaded': num_channels,
            'channel_sparsity_config': self.channel_sparsity_config,
            'cache_usage': cache_usage,
            'data_ratio': self.data_ratio,
            'sample_counts': dict(self.sample_counts),
            'train_subjects': split_subjects.get('train_subjects'),
            'val_subjects': split_subjects.get('val_subjects'),
            'test_subjects': split_subjects.get('test_subjects'),
        }

        print(f"Finetune DataLoader loaded:")
        print(f"  Dataset: {self.dataset}:{self.task}")
        if self.split_mode == 'loso' and self.held_out_subject:
            print(f"  Split: {self.split_mode}, Held-out subject: {self.held_out_subject}, Fold: {self.fold}")
        else:
            print(f"  Split: {self.split_mode}, Fold: {self.fold}")
        print(f"  Format: {self.input_format}, Patch size: {self.patch_size}")
        print(f"  Channel system: {self.channel_system}")
        if self.channel_variant:
            print(
                f"  Channel variant: {self.channel_variant} "
                f"(target={resolved_channel_count_target}, loaded={num_channels})"
            )
        print(f"  Data ratio: {self.data_ratio}")
        print(
            "  Samples: "
            f"Train {self.sample_counts.get('train')}/{self.sample_counts.get('train_original')}, "
            f"Val {self.sample_counts.get('val')}/{self.sample_counts.get('val_original')}, "
            f"Test {self.sample_counts.get('test')}/{self.sample_counts.get('test_original')}"
        )
        print(
            f"  num_classes: {num_classes}, num_channels: {num_channels}, "
            f"patch_num: {self.patch_num}, samples: {time_len}"
        )
        print(f"  ch_positions: {ch_positions}")

        # Auto-update args with dataset metadata (if provided)
        if self.args is not None:
            self.args.num_channels = num_channels
            self.args.patch_num = self.patch_num
            self.args.samples = time_len
            self.args.num_of_classes = num_classes
            self.args.ch_positions = ch_positions
            self.args.ch_names = ch_names
            self.args.model_channel_indices = model_channel_indices
            self.args.channel_variant = self.channel_variant
            self.args.channel_count_target = resolved_channel_count_target
            self.args.channel_count_loaded = num_channels
            self.args.channel_sparsity_config = self.channel_sparsity_config
            self.args.data_ratio = self.data_ratio
            self.args.sample_counts = dict(self.sample_counts)
            self.args.train_samples_original = self.sample_counts.get('train_original')
            self.args.train_samples = self.sample_counts.get('train')
            self.args.val_samples_original = self.sample_counts.get('val_original')
            self.args.val_samples = self.sample_counts.get('val')
            self.args.test_samples_original = self.sample_counts.get('test_original')
            self.args.test_samples = self.sample_counts.get('test')
            self.args.train_subjects = split_subjects.get('train_subjects')
            self.args.val_subjects = split_subjects.get('val_subjects')
            self.args.test_subjects = split_subjects.get('test_subjects')

        return data_loader

    def _patch_collate(self, batch):
        """Collate function for patch format"""
        import torch
        x_batch = torch.stack([item[0] for item in batch])
        y_batch = torch.stack([item[1] for item in batch])
        return x_batch, y_batch

    def _raw_collate(self, batch):
        """Collate function for raw format"""
        import torch
        x_batch = torch.stack([item[0] for item in batch])
        y_batch = torch.stack([item[1] for item in batch])
        return x_batch, y_batch

    def _close_dataset_handles(self, *datasets):
        seen = set()
        for dataset in datasets:
            self._close_dataset_handles_recursive(dataset, seen)

    def _close_dataset_handles_recursive(self, dataset, seen):
        if dataset is None:
            return

        dataset_id = id(dataset)
        if dataset_id in seen:
            return
        seen.add(dataset_id)

        close_fn = getattr(dataset, 'close', None)
        if callable(close_fn):
            close_fn()

        inner_dataset = getattr(dataset, 'dataset', None)
        if inner_dataset is not None:
            self._close_dataset_handles_recursive(inner_dataset, seen)

        subset = getattr(dataset, 'subset', None)
        if subset is not None:
            self._close_dataset_handles_recursive(subset, seen)
