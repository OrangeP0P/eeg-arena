import argparse
import math
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import torch
from experiments.common import resolve_device
from experiments.common.tracking import ExperimentTracker, parse_cli_kv_items
from experiments.single import apply_model_input_constraints, handle_list_variants, resolve_weight_path
from models.registry import MODEL_REGISTRY, get_model, get_model_config, get_finetune_adapter, list_models
from trainer.finetune_trainer import Trainer
from utils import setup_seed
from data.channels import PUBLIC_CHANNEL_SYSTEM_CHOICES
from data.finetune.core import resolve_loso_fold_and_subject
from data.raw_to_hdf5 import HDF5BuildParams


def parse_bool(value):
    """Parse a strict True/False CLI boolean value."""
    if isinstance(value, bool):
        return value

    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False

    raise argparse.ArgumentTypeError("Expected 'True' or 'False'.")


def parse_data_ratio(value):
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("Expected a float in (0, 1].")
    if not math.isfinite(ratio) or ratio <= 0.0 or ratio > 1.0:
        raise argparse.ArgumentTypeError("Expected a float in (0, 1].")
    return ratio


def add_common_args(parser):
    """Add common arguments"""
    parser.add_argument(
        "--model",
        type=str,
        default="cbramod",
        help=f"Model name. Available: {', '.join(sorted(list_models()))}",
    )
    parser.add_argument(
        "--use_pretrained_weights",
        type=parse_bool,
        default=True,
        metavar="{True,False}",
        help="Load pretrained weights (True/False, default: True)",
    )
    parser.add_argument(
        "--frozen",
        type=parse_bool,
        default=False,
        metavar="{True,False}",
        help="Freeze backbone parameters (True/False, default: False)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device selector: auto / cpu / cuda / cuda:N",
    )
    parser.add_argument("--cuda", type=int, default=0, help="Default CUDA index when --device is auto/cuda")
    parser.add_argument("--gpu_ids", type=str, default="1,2,3", help="GPU IDs for multi-GPU training, e.g., '0,1,2,3'")
    parser.add_argument("--use_dataparallel", action="store_true", default=False,
                        help="Use DataParallel for multi-GPU training")
    parser.add_argument("--no-use_dataparallel", dest="use_dataparallel", action="store_false")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-2, help="Weight decay")
    parser.add_argument("--optimizer", type=str, default="AdamW", help="Optimizer (AdamW, SGD)")
    parser.add_argument("--clip_value", type=float, default=1, help="Gradient clip value")
    parser.add_argument("--num_workers", type=int, default=16, help="DataLoader workers")

    # Dataset arguments
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (e.g. 'bcic_iv_2a'). "
                             "See datasets/__init__.py for all registered datasets.")
    parser.add_argument("--task", type=str, required=True,
                        help="Task name within the dataset (e.g. 'motor_imagery'). "
                             "See datasets/__init__.py for all registered tasks.")
    parser.add_argument("--cache_dir", type=str, required=True,
                        help="Base directory for HDF5 cache and locator CSV files. "
                            "Subdirectories are created automatically per dataset/task.")
    parser.add_argument("--split_mode", type=str, default="cs",
                        help="Split mode (cs/ws/loso/cross_trial)")
    parser.add_argument("--fold", type=int, default=4,
                        help="Fold index (1-5 for cs/ws, 1 to num_subjects for loso, 1 for cross_trial)")
    parser.add_argument(
        "--held_out_subject",
        "--held-out-subject",
        dest="held_out_subject",
        type=str,
        default=None,
        help="Optional LOSO held-out subject id. When set, it takes precedence over --fold.",
    )
    parser.add_argument("--input_format", type=str, default='patched',
                       choices=['patched', 'raw'],
                       help="Input data format: 'patched' for (C, num_patches, patch_size), 'raw' for (C, T). "
                            "See the model library in README.md for each model's expected input format.")
    parser.add_argument("--patch_size", type=int, default=200,
                       help="Patch size for patched format (default: 200 samples = 1s at 200Hz). "
                            "See the model library in README.md for model input details.")
    parser.add_argument("--channel_system", type=str, default='10-20',
                       choices=list(PUBLIC_CHANNEL_SYSTEM_CHOICES),
                       help="EEG channel system for data loading (10-20, 10-10, 10-5, dynamic, custom)")
    parser.add_argument(
        "--channel_variant",
        "--channel_sparsity_variant",
        dest="channel_variant",
        type=str,
        default=None,
        help="Optional channel sparsity subset id. Only valid when --channel_system custom.",
    )
    parser.add_argument(
        "--channel_count_target",
        type=int,
        default=None,
        help="Optional expected user-space channel count for a custom channel variant.",
    )
    parser.add_argument(
        "--channel_sparsity_config",
        type=str,
        default=None,
        help="Optional JSON file defining channel sparsity variants for --channel_system custom.",
    )
    parser.add_argument(
        "--data_ratio",
        "--data-ratio",
        dest="data_ratio",
        type=parse_data_ratio,
        default=1.0,
        help="Randomly keep this fraction of each train/val/test split after splitting. Must be in (0, 1].",
    )

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./experiments/output/debug_run",
                        help="Output root directory. Per-run artifacts are saved to <output_dir>/runs/<run_id>/")
    parser.add_argument("--save_checkpoints", "--save-checkpoints", dest="save_checkpoints",
                        action="store_true", default=False,
                        help="Save best/last model checkpoints under the run directory.")
    parser.add_argument("--no-save_checkpoints", "--no-save-checkpoints", dest="save_checkpoints",
                        action="store_false")

    # Training arguments
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing")
    parser.add_argument("--multi_lr", action="store_true", default=True,
                        help="Use different LR for backbone/head")
    parser.add_argument("--no-multi_lr", "--no-multi-lr", dest="multi_lr", action="store_false")
    parser.add_argument("--encoder_lr_scale", type=float, default=1.0,
                        help="Learning rate scale for encoder/backbone")
    parser.add_argument("--classifier_lr_scale", type=float, default=1.0,
                        help="Learning rate scale for classifier/head")
    parser.add_argument(
        "--early_stopping_patience",
        "--early-stopping-patience",
        dest="early_stopping_patience",
        type=int,
        default=10,
        help="Stop training after this many consecutive epochs without validation-metric improvement. "
             "Set <= 0 to disable early stopping.",
    )

    # Variant listing
    parser.add_argument("--list_variants", action="store_true", default=False,
                        help="List available weight variants for the selected model and exit")

    # ---- HDF5 build parameters (used when building HDF5 from raw data) ----
    parser.add_argument(
        "--pipeline",
        type=str,
        default=None,
        metavar="PIPELINE_PY",
        help="Path to a preprocessing pipeline .py file. "
             "No-param pipelines export 'pipeline(raw) -> raw'. "
             "Parameterised pipelines export 'make_pipeline(**kwargs) -> pipeline(raw) -> raw' "
             "and require --pipeline_param. "
             "Built-in presets in datasets/pipelines/: default.py, no_normalize.py, raw_only.py, "
             "classical_pipeline_with_z_score.py. "
             "Default: None (no preprocessing).",
    )
    parser.add_argument(
        "--pipeline_param",
        type=str,
        nargs='+',
        default=None,
        metavar="KEY=VALUE",
        help="Parameters forwarded to make_pipeline() in the pipeline file, e.g. "
             "--pipeline_param resample=200 lower_bound=0.1 upper_bound=75 notch=50. "
             "Every key must match a parameter of make_pipeline(); unknown keys raise an error. "
             "Only valid when --pipeline points to a file with make_pipeline().",
    )
    parser.add_argument(
        "--epoch_param",
        type=str,
        nargs='+',
        default=None,
        metavar="KEY=VALUE",
        help="Override epoch parameters, e.g. --epoch_param tmin=-0.5 tmax=4.0. "
             "Valid keys depend on epoch_method: "
             "epoch_by_event_hdf5 → tmin, tmax; "
             "epoch_by_segmentation_hdf5 → segment_length, overlap; "
             "epoch_by_long_event_hdf5 → tmin, tmax, overlap. "
             "Defaults to the task's default_epoch_params if not specified.",
    )
    parser.add_argument(
        "--rebuild_hdf5",
        action="store_true",
        default=False,
        help="Force rebuild of the HDF5 cache even if it already exists.",
    )


def print_run_configuration(params):
    """Print a concise run summary before training starts."""
    print(f"\n{'=' * 50}")
    print(f"Experiment ID: {params.exp_id}")
    print(f"Run ID: {params.run_id}")
    print(f"Run directory: {params.run_dir}")
    print(f"Model: {params.model}")
    print(f"Device: {params.device}")
    if params.use_dataparallel:
        print(f"Multi-GPU: DataParallel with GPUs [{params.gpu_ids}]")
    print(f"Dataset: {params.dataset}:{params.task}")
    if params.split_mode == "loso":
        print(f"Split mode: {params.split_mode}, Held-out subject: {params.held_out_subject}, Fold: {params.fold}")
    else:
        print(f"Split mode: {params.split_mode}, Fold: {params.fold}")
    print(f"Channel system: {params.channel_system}")
    if getattr(params, "channel_variant", None):
        print(f"Channel variant: {params.channel_variant}")
    print(f"Data ratio: {params.data_ratio}")
    print(f"Epochs: {params.epochs}, Batch: {params.batch_size}, LR: {params.lr}")
    if params.early_stopping_patience > 0:
        print(f"Early stopping patience: {params.early_stopping_patience}")
    else:
        print("Early stopping: disabled")
    print(f"{'=' * 50}\n")


def resolve_split_args(params):
    if params.split_mode == "cross_trial":
        params.fold = 1
        params.held_out_subject = None
        return
    if params.split_mode != "loso":
        return

    build_params = HDF5BuildParams(
        epoch_params=parse_cli_kv_items(getattr(params, "epoch_param", None)),
        pipeline=getattr(params, "pipeline", None),
        pipeline_params=parse_cli_kv_items(getattr(params, "pipeline_param", None)),
        rebuild_hdf5=False,
        num_workers=getattr(params, "num_workers", 16),
    )
    buffer = StringIO()
    with redirect_stdout(buffer), redirect_stderr(buffer):
        resolved_fold, resolved_subject = resolve_loso_fold_and_subject(
            params.dataset,
            params.task,
            fold=params.fold,
            held_out_subject=params.held_out_subject,
            build_params=build_params,
            cache_dir=params.cache_dir,
        )
    params.fold = resolved_fold
    params.held_out_subject = resolved_subject


def main():
    parser = argparse.ArgumentParser(description="Unified EEG Model Finetune")

    # 1. Add common arguments
    add_common_args(parser)

    # 2. Dynamically add model-specific arguments
    # Parse once to get model name
    args, _ = parser.parse_known_args()

    # Check if model is registered
    if args.model.lower() not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {args.model}. Available models: {list_models()}"
        )

    # Add model-specific arguments
    model_config = get_model_config(args.model)
    model_config.add_args(parser, mode='finetune')

    # 3. Full parse arguments
    params = parser.parse_args()
    if params.channel_system == "custom" and not params.channel_variant:
        parser.error("--channel_system custom requires --channel_variant or --channel_sparsity_variant.")
    if params.channel_system != "custom" and params.channel_variant is not None:
        parser.error("--channel_variant/--channel_sparsity_variant is only valid when --channel_system custom.")
    if params.channel_system != "custom" and params.channel_count_target is not None:
        parser.error("--channel_count_target is only valid when --channel_system custom.")
    if params.channel_system != "custom" and params.channel_sparsity_config is not None:
        parser.error("--channel_sparsity_config is only valid when --channel_system custom.")
    apply_model_input_constraints(parser, params, model_config, sys.argv[1:])
    params.output_dir = os.path.abspath(params.output_dir)

    if handle_list_variants(params, model_config):
        return

    # Resolve --variant to foundation_dir path
    resolve_weight_path(params, model_config, parser)

    if hasattr(params, 'foundation_dir') and getattr(params, 'foundation_dir', None):
        params.foundation_dir = os.path.abspath(params.foundation_dir)

    resolve_split_args(params)

    resolved_device = resolve_device(params.device, params.cuda, params.gpu_ids, params.use_dataparallel)
    params.device_obj = resolved_device
    params.device = str(resolved_device)

    tracker = None

    try:
        tracker = ExperimentTracker(os.path.dirname(os.path.abspath(__file__)), params)
        tracker.attach_run_paths(params)
        tracker.install_console_tee()
        tracker.write_config_snapshot(params)

        if resolved_device.type == "cuda":
            torch.cuda.set_device(resolved_device.index if resolved_device.index is not None else params.cuda)

        # 4. Print configuration
        print_run_configuration(params)

        # 5. Setup environment
        setup_seed(params.seed)

        # 6. Load data
        print(f"Loading dataset: {params.dataset}:{params.task}...")
        adapter_class = get_finetune_adapter(params.model)
        if adapter_class is None:
            raise ValueError(f"No dataset adapter found for model: {params.model}")
        data_loader = adapter_class(params).get_data_loader()
        cache_usage = data_loader.get("cache_usage")
        split_subject_fields = {
            "train_subjects": data_loader.get("train_subjects"),
            "val_subjects": data_loader.get("val_subjects"),
            "test_subjects": data_loader.get("test_subjects"),
        }
        sample_counts = data_loader.get("sample_counts") or {}
        sample_count_fields = {
            "data_ratio": data_loader.get("data_ratio", getattr(params, "data_ratio", 1.0)),
            "sample_counts": sample_counts,
            "train_samples_original": sample_counts.get("train_original"),
            "train_samples": sample_counts.get("train"),
            "val_samples_original": sample_counts.get("val_original"),
            "val_samples": sample_counts.get("val"),
            "test_samples_original": sample_counts.get("test_original"),
            "test_samples": sample_counts.get("test"),
            "channel_count_target": data_loader.get("channel_count_target"),
            "channel_count_loaded": data_loader.get("channel_count_loaded"),
        }
        params.train_subjects = split_subject_fields["train_subjects"]
        params.val_subjects = split_subject_fields["val_subjects"]
        params.test_subjects = split_subject_fields["test_subjects"]
        for key, value in sample_count_fields.items():
            setattr(params, key, value)

        status_update_fields = dict(split_subject_fields)
        status_update_fields.update(sample_count_fields)
        if cache_usage:
            params.cache_usage = cache_usage
            status_update_fields["cache_usage"] = cache_usage
        tracker.update_status(**status_update_fields)
        print(f"Data loaded: Train={len(data_loader['train'])}, Val={len(data_loader['val'])}, Test={len(data_loader['test'])}")
        # Note: params.num_channels, params.patch_num, params.num_of_classes are auto-updated by FinetuneDataLoader

        params.config_id = tracker.ensure_config_id(params)
        print(f"Config ID: {params.config_id}")

        # 7. Create model
        print(f"Creating model: {params.model}...")
        model_class = get_model(params.model)
        model = model_class(params)
        print(model)

        # 8. Train
        print("Starting training...")
        tracker.mark_running()
        trainer = Trainer(model, data_loader, params, tracker=tracker)
        trainer.train()
        tracker.mark_completed()
        tracker.append_experiment_row(params)

        print("\nDone!")
    except Exception as exc:
        if tracker is not None:
            tracker.record_failure(params, exc)
        raise
    finally:
        if tracker is not None:
            tracker.restore_console_streams()

if __name__ == "__main__":
    main()
