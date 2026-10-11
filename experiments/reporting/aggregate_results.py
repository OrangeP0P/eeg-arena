#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.common.runtime_metadata import (
    clean_optional_text,
    extract_final_metric_fields,
    subject_list_text,
)
from experiments.common.registry_views import canonical_experiments_csv_path
from experiments.common.result_schema import EXPERIMENTS_CSV_HEADERS


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Aggregate EEG-Arena run results from registry or a batch suite.")
    parser.add_argument("--registry-dir", type=str, default=None, help="Path to experiments/experiment_registry")
    parser.add_argument("--suite-dir", type=str, default=None, help="Path to one batch output directory")
    parser.add_argument(
        "--source",
        type=str,
        default="auto",
        choices=["registry", "suite", "auto"],
        help="Primary input source",
    )
    parser.add_argument("--output", type=str, default=None, help="Write output to this file instead of stdout")
    parser.add_argument(
        "--format",
        type=str,
        default="csv",
        choices=["csv", "jsonl"],
        help="Output format",
    )
    return parser.parse_args(argv)


def read_csv_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_csv_headers(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or [])


def read_json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def output_path_from_relpath(suite_dir: Path, relpath: str | None):
    relpath = clean_optional_text(relpath)
    if relpath is None:
        return None
    return str((suite_dir / relpath).resolve())


def normalize_run_status(status: str | None):
    status = clean_optional_text(status)
    mapping = {
        "Pending": "pending",
        "Running": "running",
        "Success": "completed",
        "Failed": "failed",
    }
    return mapping.get(status, status)


def suite_row_to_experiment_row(row, suite_dir: Path):
    suite_name = suite_dir.name
    run_id = clean_optional_text(row.get("run_id"))
    run_relpath = clean_optional_text(row.get("run_relpath")) or (f"runs/{run_id}" if run_id else None)
    config_relpath = clean_optional_text(row.get("config_relpath")) or (f"runs/{run_id}/config.json" if run_id else None)
    status_relpath = clean_optional_text(row.get("status_relpath")) or (f"runs/{run_id}/status.json" if run_id else None)
    final_relpath = clean_optional_text(row.get("final_test_metrics_relpath")) or (
        f"runs/{run_id}/metrics/final_test_metrics.json" if run_id else None
    )
    log_relpath = clean_optional_text(row.get("log_relpath")) or (f"runs/{run_id}/logs/console.log" if run_id else None)

    config_path = output_path_from_relpath(suite_dir, config_relpath)
    status_path = output_path_from_relpath(suite_dir, status_relpath)
    final_test_metrics_path = output_path_from_relpath(suite_dir, final_relpath)
    log_path = output_path_from_relpath(suite_dir, log_relpath)

    config_payload = read_json(Path(config_path)) if config_path else None
    status_payload = read_json(Path(status_path)) if status_path else None
    final_test_payload = read_json(Path(final_test_metrics_path)) if final_test_metrics_path else None
    resolved_params = (config_payload or {}).get("resolved_params") or {}
    status_selection_name = (status_payload or {}).get("selection_metric_name")
    status_selection_value = (status_payload or {}).get("selection_metric_value")
    final_metric_fields = extract_final_metric_fields(final_test_payload)
    sample_counts = (status_payload or {}).get("sample_counts") or {}

    result = {field: None for field in EXPERIMENTS_CSV_HEADERS}
    result.update(
        {
            "exp_id": clean_optional_text(row.get("exp_id")),
            "run_id": run_id,
            "config_id": clean_optional_text(row.get("config_id")),
            "data_config_id": clean_optional_text(row.get("data_config_id")),
            "data_config_key": clean_optional_text(row.get("data_config_key")),
            "suite_name": suite_name,
            "suite_dir": str(suite_dir),
            "run_dir": output_path_from_relpath(suite_dir, run_relpath),
            "model": clean_optional_text(row.get("model")),
            "dataset": clean_optional_text(row.get("dataset")),
            "task": clean_optional_text(row.get("task")),
            "split_mode": clean_optional_text(row.get("split_mode")),
            "held_out_subject": clean_optional_text(row.get("held_out_subject")) or clean_optional_text((status_payload or {}).get("held_out_subject")),
            "fold": clean_optional_text(row.get("fold")),
            "train_subjects": subject_list_text(row.get("train_subjects")) or subject_list_text((status_payload or {}).get("train_subjects")),
            "val_subjects": subject_list_text(row.get("val_subjects")) or subject_list_text((status_payload or {}).get("val_subjects")),
            "test_subjects": subject_list_text(row.get("test_subjects")) or subject_list_text((status_payload or {}).get("test_subjects")),
            "seed": clean_optional_text(row.get("seed")),
            "channel_system": clean_optional_text(row.get("channel_system")),
            "input_format": clean_optional_text(row.get("input_format")) or clean_optional_text(resolved_params.get("input_format")),
            "variant": clean_optional_text(row.get("variant")) or clean_optional_text(resolved_params.get("variant")),
            "epochs": clean_optional_text(row.get("epochs")) or clean_optional_text(resolved_params.get("epochs")),
            "batch_size": clean_optional_text(row.get("batch_size")) or clean_optional_text(resolved_params.get("batch_size")),
            "data_ratio": (
                clean_optional_text(row.get("data_ratio"))
                or clean_optional_text((status_payload or {}).get("data_ratio"))
                or clean_optional_text(resolved_params.get("data_ratio"))
            ),
            "train_samples_original": clean_optional_text(row.get("train_samples_original")) or clean_optional_text((status_payload or {}).get("train_samples_original")) or clean_optional_text(sample_counts.get("train_original")),
            "train_samples": clean_optional_text(row.get("train_samples")) or clean_optional_text((status_payload or {}).get("train_samples")) or clean_optional_text(sample_counts.get("train")),
            "val_samples_original": clean_optional_text(row.get("val_samples_original")) or clean_optional_text((status_payload or {}).get("val_samples_original")) or clean_optional_text(sample_counts.get("val_original")),
            "val_samples": clean_optional_text(row.get("val_samples")) or clean_optional_text((status_payload or {}).get("val_samples")) or clean_optional_text(sample_counts.get("val")),
            "test_samples_original": clean_optional_text(row.get("test_samples_original")) or clean_optional_text((status_payload or {}).get("test_samples_original")) or clean_optional_text(sample_counts.get("test_original")),
            "test_samples": clean_optional_text(row.get("test_samples")) or clean_optional_text((status_payload or {}).get("test_samples")) or clean_optional_text(sample_counts.get("test")),
            "lr": clean_optional_text(resolved_params.get("lr")),
            "encoder_lr_scale": clean_optional_text(resolved_params.get("encoder_lr_scale")),
            "classifier_lr_scale": clean_optional_text(row.get("classifier_lr_scale")) or clean_optional_text(resolved_params.get("classifier_lr_scale")),
            "epoch_method": clean_optional_text(row.get("epoch_method")) or clean_optional_text((status_payload or {}).get("epoch_method")),
            "pipeline_key": clean_optional_text(row.get("pipeline_key")) or clean_optional_text((status_payload or {}).get("pipeline_key")),
            "pipeline": clean_optional_text(row.get("pipeline")) or clean_optional_text(resolved_params.get("pipeline")),
            "pipeline_params": clean_optional_text(row.get("pipeline_params")),
            "epoch_params": clean_optional_text(row.get("epoch_params")),
            "cache_root": clean_optional_text(row.get("cache_root")),
            "locator_cache_id": clean_optional_text(row.get("locator_cache_id")),
            "hdf5_cache_id": clean_optional_text(row.get("hdf5_cache_id")),
            "status": normalize_run_status(clean_optional_text((status_payload or {}).get("status")) or row.get("status")),
            "best_epoch": clean_optional_text(row.get("best_epoch")) or clean_optional_text((status_payload or {}).get("best_epoch")),
            "selection_metric_name": clean_optional_text(row.get("selection_metric_name")) or clean_optional_text(status_selection_name),
            "selection_metric_value": clean_optional_text(row.get("selection_metric_value")) or clean_optional_text(status_selection_value),
            "final_acc": clean_optional_text(row.get("final_acc")) or final_metric_fields.get("final_acc"),
            "final_f1_macro": clean_optional_text(row.get("final_f1_macro")) or final_metric_fields.get("final_f1_macro"),
            "final_f1_weighted": clean_optional_text(row.get("final_f1_weighted")) or final_metric_fields.get("final_f1_weighted"),
            "final_kappa": clean_optional_text(row.get("final_kappa")) or final_metric_fields.get("final_kappa"),
            "started_at": clean_optional_text((status_payload or {}).get("started_at")) or clean_optional_text(row.get("started_at")),
            "finished_at": clean_optional_text((status_payload or {}).get("finished_at")) or clean_optional_text(row.get("finished_at")),
            "duration_sec": clean_optional_text((status_payload or {}).get("duration_sec")) or clean_optional_text(row.get("duration_sec")),
            "config_path": config_path,
            "status_path": status_path,
            "final_test_metrics_path": final_test_metrics_path,
            "log_path": log_path,
        }
    )
    return result


def load_registry_rows(registry_dir: Path):
    experiments_path = Path(canonical_experiments_csv_path(str(registry_dir)))
    if not experiments_path.exists():
        fallback_path = registry_dir / "experiments.csv"
        if read_csv_headers(fallback_path) == EXPERIMENTS_CSV_HEADERS:
            experiments_path = fallback_path
    if not experiments_path.exists():
        raise FileNotFoundError(
            f"Canonical registry experiments file not found under {registry_dir / 'internal'}. "
            "Human-readable experiments.csv is not accepted as the registry source."
        )
    rows = read_csv_rows(experiments_path)
    return [{field: row.get(field) for field in EXPERIMENTS_CSV_HEADERS} for row in rows]


def load_suite_rows(suite_dir: Path):
    task_status_path = suite_dir / "scheduler" / "task_status.csv"
    if not task_status_path.exists():
        raise FileNotFoundError(f"Suite task_status.csv not found: {task_status_path}")

    rows = []
    for row in read_csv_rows(task_status_path):
        status = clean_optional_text(row.get("status")) or "Pending"
        if status == "Pending" and not clean_optional_text(row.get("run_id")) and not clean_optional_text(row.get("exp_id")):
            continue
        rows.append(suite_row_to_experiment_row(row, suite_dir))
    return rows


def choose_source(args):
    registry_dir = Path(args.registry_dir).resolve() if args.registry_dir else None
    suite_dir = Path(args.suite_dir).resolve() if args.suite_dir else None

    if args.source == "registry":
        if registry_dir is None:
            raise ValueError("--source registry requires --registry-dir")
        return "registry", registry_dir

    if args.source == "suite":
        if suite_dir is None:
            raise ValueError("--source suite requires --suite-dir")
        return "suite", suite_dir

    registry_canonical_path = Path(canonical_experiments_csv_path(str(registry_dir))) if registry_dir is not None else None
    if registry_canonical_path is not None and registry_canonical_path.exists():
        return "registry", registry_dir
    if registry_dir is not None and read_csv_headers(registry_dir / "experiments.csv") == EXPERIMENTS_CSV_HEADERS:
        return "registry", registry_dir
    if suite_dir is not None and (suite_dir / "scheduler" / "task_status.csv").exists():
        return "suite", suite_dir
    raise ValueError("Could not auto-detect a valid source. Provide --registry-dir or --suite-dir.")


def write_rows(rows, output_path: Path | None, output_format: str):
    if output_path is None:
        if output_format == "csv":
            writer = csv.DictWriter(sys.stdout, fieldnames=EXPERIMENTS_CSV_HEADERS)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
            return
        for row in rows:
            sys.stdout.write(json.dumps(row, ensure_ascii=False) + "\n")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "csv":
        with output_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=EXPERIMENTS_CSV_HEADERS)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return

    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv=None):
    args = parse_args(argv)
    source, source_path = choose_source(args)
    if source == "registry":
        rows = load_registry_rows(source_path)
    else:
        rows = load_suite_rows(source_path)

    output_path = Path(args.output).resolve() if args.output else None
    write_rows(rows, output_path, args.format)


if __name__ == "__main__":
    main()
