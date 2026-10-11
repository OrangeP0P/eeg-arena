import csv
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch

from experiments.common.registry_views import (
    _gpfs_exclusive_lock,
    build_registry_layout,
    ensure_registry_layout,
    refresh_registry_views,
)
from experiments.common.result_schema import EXPERIMENTS_CSV_HEADERS
from experiments.common.runtime_metadata import (
    clean_optional_text,
    extract_final_metric_fields,
    resolve_data_config_fields,
    subject_list_text,
)

CONFIG_ID_EXCLUDE_KEYS = {
    "output_dir",
    "run_dir",
    "model_dir",
    "log_dir",
    "metrics_dir",
    "checkpoint_dir",
    "logs_dir",
    "console_log_path",
    "device_obj",
    "device",
    "cuda",
    "gpu_ids",
    "exp_id",
    "run_id",
    "config_id",
    "start_time",
    "end_time",
    "experiment_tracker",
    "cache_usage",
    "sample_counts",
    "train_samples_original",
    "train_samples",
    "val_samples_original",
    "val_samples",
    "test_samples_original",
    "test_samples",
    "train_subjects",
    "val_subjects",
    "test_subjects",
    "ch_names",
    "ch_positions",
    "num_channels",
    "num_of_classes",
    "model_channel_indices",
    "patch_num",
    "channel_count_loaded",
}

SNAPSHOT_EXCLUDE_KEYS = {"experiment_tracker", "cache_usage"}


def now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _is_torch_device_instance(value):
    device_type = getattr(torch, "device", None)
    return isinstance(device_type, type) and isinstance(value, device_type)


def _json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    if _is_torch_device_instance(obj):
        return str(obj)
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except Exception:
            pass
    if hasattr(obj, "tolist") and callable(obj.tolist):
        return obj.tolist()
    if isinstance(obj, set):
        return sorted(obj)
    return str(obj)


def normalize_for_json(value):
    if isinstance(value, dict):
        return {
            str(key): normalize_for_json(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [normalize_for_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if _is_torch_device_instance(value):
        return str(value)
    if hasattr(value, "item") and callable(value.item):
        try:
            return normalize_for_json(value.item())
        except Exception:
            pass
    if hasattr(value, "tolist") and callable(value.tolist):
        return normalize_for_json(value.tolist())
    if isinstance(value, set):
        return [normalize_for_json(item) for item in sorted(value)]
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return str(value)
    return value


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(normalize_for_json(payload), f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp_path, path)


def append_jsonl(path, record):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(normalize_for_json(record), ensure_ascii=False, default=_json_default) + "\n")


def scalarize_metrics(metrics):
    scalar_metrics = {}
    for key, value in metrics.items():
        if key == "confusion_matrix":
            continue
        normalized_value = normalize_for_json(value)
        if isinstance(normalized_value, (int, float, bool, str)) or normalized_value is None:
            scalar_metrics[key] = normalized_value
    return scalar_metrics


def parse_param_scalar(value):
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


def parse_cli_kv_items(raw_items):
    if raw_items is None:
        return None

    if isinstance(raw_items, dict):
        return {
            str(key): parse_param_scalar(value)
            for key, value in raw_items.items()
        }

    parsed = {}
    for item in raw_items:
        if '=' not in item:
            raise ValueError(f"Expected key=value item, got '{item}'.")
        key, value = item.split('=', 1)
        parsed[key] = parse_param_scalar(value)
    return parsed


def stable_json_text(value):
    if value is None:
        return None
    return json.dumps(normalize_for_json(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def resolve_run_metadata_fields(params):
    cache_usage = getattr(params, "cache_usage", None) or {}
    hdf5_usage = cache_usage.get("hdf5") or {}

    dataset = cache_usage.get("dataset") or getattr(params, "dataset", None) or getattr(params, "downstream_dataset", None)
    task = cache_usage.get("task") or getattr(params, "task", None)
    epoch_method = clean_optional_text(cache_usage.get("epoch_method"))
    pipeline_key = clean_optional_text(cache_usage.get("pipeline_key"))

    pipeline = cache_usage.get("pipeline", getattr(params, "pipeline", None))
    pipeline_params = cache_usage.get("pipeline_params")
    if pipeline_params is None:
        pipeline_params = parse_cli_kv_items(getattr(params, "pipeline_param", None))

    epoch_params = cache_usage.get("epoch_params")
    if epoch_params is None:
        epoch_params = parse_cli_kv_items(getattr(params, "epoch_param", None))

    data_config_fields = resolve_data_config_fields(
        dataset=dataset,
        task=task,
        epoch_method=epoch_method,
        pipeline_key=pipeline_key,
        pipeline_params=pipeline_params,
        epoch_params=epoch_params,
        data_config_key=cache_usage.get("data_config_key") or hdf5_usage.get("config_key"),
        data_config_id=cache_usage.get("data_config_id"),
    )

    return {
        "dataset": dataset,
        "task": task,
        "epoch_method": epoch_method,
        "pipeline_key": pipeline_key,
        "pipeline": pipeline,
        "pipeline_params": stable_json_text(pipeline_params),
        "epoch_params": stable_json_text(epoch_params),
        **data_config_fields,
    }


def resolve_cache_link_fields(params):
    cache_usage = getattr(params, "cache_usage", None) or {}
    locator_usage = cache_usage.get("locator") or {}
    hdf5_usage = cache_usage.get("hdf5") or {}

    return {
        "cache_root": cache_usage.get("cache_root"),
        "locator_cache_id": locator_usage.get("cache_id"),
        "hdf5_cache_id": hdf5_usage.get("cache_id"),
    }


def load_final_metric_fields(final_test_metrics_path):
    if not final_test_metrics_path or not os.path.exists(final_test_metrics_path):
        return extract_final_metric_fields(None)

    try:
        with open(final_test_metrics_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return extract_final_metric_fields(None)
    return extract_final_metric_fields(payload)


def resolve_status_fields_from_cache_usage(cache_usage):
    cache_usage = cache_usage or {}
    hdf5_usage = cache_usage.get("hdf5") or {}
    epoch_method = clean_optional_text(cache_usage.get("epoch_method"))
    pipeline_key = clean_optional_text(cache_usage.get("pipeline_key"))
    pipeline_params = cache_usage.get("pipeline_params")
    epoch_params = cache_usage.get("epoch_params")
    data_config_fields = resolve_data_config_fields(
        dataset=cache_usage.get("dataset"),
        task=cache_usage.get("task"),
        epoch_method=epoch_method,
        pipeline_key=pipeline_key,
        pipeline_params=pipeline_params,
        epoch_params=epoch_params,
        data_config_key=cache_usage.get("data_config_key") or hdf5_usage.get("config_key"),
        data_config_id=cache_usage.get("data_config_id"),
    )
    return {
        "task": cache_usage.get("task"),
        "epoch_method": epoch_method,
        "pipeline_key": pipeline_key,
        **data_config_fields,
    }


def _sanitize_run_token(value):
    text = clean_optional_text(value)
    if text is None:
        return "default"
    safe_chars = []
    for char in text:
        if char.isalnum() or char in {"-", "_"}:
            safe_chars.append(char)
        elif char == ".":
            safe_chars.append("p")
        else:
            safe_chars.append("_")
    return "".join(safe_chars) or "default"


def _format_split_fragment(split_mode, fold, held_out_subject=None):
    split_mode = getattr(split_mode, "strip", lambda: split_mode)()
    held_out_subject = clean_optional_text(held_out_subject)
    if split_mode == "loso" and held_out_subject is not None:
        return f"{split_mode}_s{_sanitize_run_token(held_out_subject)}"
    return f"{split_mode}_f{fold}"


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams
        self.encoding = getattr(streams[0], "encoding", "utf-8") if streams else "utf-8"

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)

    def fileno(self):
        return self.streams[0].fileno()

    def close(self):
        self.flush()


class ExperimentTracker:
    def __init__(self, repo_root, params):
        self.repo_root = os.path.abspath(repo_root)
        self.layout = build_registry_layout(self.repo_root)
        ensure_registry_layout(self.layout)
        self.registry_dir = self.layout.registry_dir
        self.next_exp_id_path = self.layout.next_exp_id_path
        self.next_exp_id_lock_path = self.layout.next_exp_id_lock_path
        self.registry_lock_path = self.layout.registry_lock_path

        self.exp_id = self._allocate_exp_id()
        self.run_id = self._build_run_id(params)
        self.suite_dir = os.path.abspath(getattr(params, "output_dir"))
        self.suite_name = os.path.basename(os.path.normpath(self.suite_dir))
        self.run_dir = os.path.join(self.suite_dir, "runs", self.run_id)
        self.metrics_dir = os.path.join(self.run_dir, "metrics")
        self.checkpoint_dir = os.path.join(self.run_dir, "checkpoints")
        self.logs_dir = os.path.join(self.run_dir, "logs")
        self.save_checkpoints = bool(getattr(params, "save_checkpoints", True))
        self.console_log_path = os.path.join(self.logs_dir, "console.log")
        self.config_path = os.path.join(self.run_dir, "config.json")
        self.status_path = os.path.join(self.run_dir, "status.json")
        self.epoch_metrics_path = os.path.join(self.metrics_dir, "epoch_metrics.jsonl")
        self.best_val_metrics_path = os.path.join(self.metrics_dir, "best_val_metrics.json")
        self.final_test_metrics_path = os.path.join(self.metrics_dir, "final_test_metrics.json")
        self.best_checkpoint_path = (
            os.path.join(self.checkpoint_dir, "best.pt") if self.save_checkpoints else None
        )
        self.last_checkpoint_path = (
            os.path.join(self.checkpoint_dir, "last.pt") if self.save_checkpoints else None
        )
        self.experiments_csv_path = self.layout.experiments_canonical_path
        self.configs_jsonl_path = self.layout.configs_jsonl_path

        self.config_id = None
        self.canonical_config = None
        self._stdout = None
        self._stderr = None
        self._console_file = None
        self._experiment_row_written = False

        self.status = {
            "exp_id": self.exp_id,
            "run_id": self.run_id,
            "config_id": self.config_id,
            "model": getattr(params, "model", None),
            "dataset": getattr(params, "dataset", None) or getattr(params, "downstream_dataset", None),
            "task": getattr(params, "task", None),
            "split_mode": getattr(params, "split_mode", None),
            "held_out_subject": getattr(params, "held_out_subject", None),
            "fold": getattr(params, "fold", None),
            "channel_system": getattr(params, "channel_system", None),
            "channel_variant": getattr(params, "channel_variant", None),
            "channel_sparsity_config": getattr(params, "channel_sparsity_config", None),
            "channel_count_target": getattr(params, "channel_count_target", None),
            "channel_count_loaded": getattr(params, "channel_count_loaded", None),
            "train_subjects": getattr(params, "train_subjects", None),
            "val_subjects": getattr(params, "val_subjects", None),
            "test_subjects": getattr(params, "test_subjects", None),
            "data_ratio": getattr(params, "data_ratio", 1.0),
            "sample_counts": getattr(params, "sample_counts", None),
            "train_samples_original": getattr(params, "train_samples_original", None),
            "train_samples": getattr(params, "train_samples", None),
            "val_samples_original": getattr(params, "val_samples_original", None),
            "val_samples": getattr(params, "val_samples", None),
            "test_samples_original": getattr(params, "test_samples_original", None),
            "test_samples": getattr(params, "test_samples", None),
            "data_config_id": None,
            "data_config_key": None,
            "epoch_method": None,
            "pipeline_key": None,
            "status": "initializing",
            "started_at": now_iso(),
            "finished_at": None,
            "duration_sec": None,
            "best_epoch": None,
            "selection_metric_name": None,
            "selection_metric_value": None,
            "best_checkpoint_path": self.best_checkpoint_path,
            "last_checkpoint_path": self.last_checkpoint_path,
            "final_test_metrics_path": self.final_test_metrics_path,
            "error_message": None,
            "log_path": self.console_log_path,
            "cache_usage": None,
        }

        for directory in (
            self.registry_dir,
            self.suite_dir,
            self.run_dir,
            self.metrics_dir,
            self.logs_dir,
        ):
            os.makedirs(directory, exist_ok=True)

        if self.save_checkpoints:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.write_status()

    def _allocate_exp_id(self):
        os.makedirs(self.registry_dir, exist_ok=True)
        with _gpfs_exclusive_lock(self.next_exp_id_lock_path):
            next_exp_id = 1
            if os.path.exists(self.next_exp_id_path):
                with open(self.next_exp_id_path, "r", encoding="utf-8") as f:
                    raw_value = f.read().strip()
                if raw_value:
                    next_exp_id = int(raw_value)

            allocated_exp_id = next_exp_id
            with open(self.next_exp_id_path, "w", encoding="utf-8") as f:
                f.write(str(allocated_exp_id + 1))

            return allocated_exp_id

    def _build_run_id(self, params):
        dataset_str = getattr(params, 'dataset', None) or getattr(params, 'downstream_dataset', 'unknown')
        task_str = getattr(params, 'task', None)
        if task_str:
            dataset_str = f"{dataset_str}__{task_str}"
        split_fragment = _format_split_fragment(
            getattr(params, "split_mode"),
            getattr(params, "fold"),
            getattr(params, "held_out_subject", None),
        )
        return (
            f"{self.exp_id:06d}__{getattr(params, 'model')}__"
            f"{dataset_str}__"
            f"{split_fragment}__"
            f"seed{getattr(params, 'seed')}"
        )

    def attach_run_paths(self, params):
        """Populate run-specific output paths on argparse params."""
        params.exp_id = self.exp_id
        params.run_id = self.run_id
        params.config_id = self.config_id
        params.run_dir = self.run_dir
        params.metrics_dir = self.metrics_dir
        params.checkpoint_dir = self.checkpoint_dir
        params.logs_dir = self.logs_dir
        params.console_log_path = self.console_log_path

        # Backward-compatible aliases for older code paths.
        params.model_dir = self.checkpoint_dir
        params.log_dir = self.logs_dir

    def install_console_tee(self):
        if self._console_file is not None:
            return

        self._stdout = sys.stdout
        self._stderr = sys.stderr
        self._console_file = open(self.console_log_path, "a", encoding="utf-8", buffering=1)
        sys.stdout = TeeStream(self._stdout, self._console_file)
        sys.stderr = TeeStream(self._stderr, self._console_file)

    def restore_console_streams(self):
        if self._console_file is None:
            return

        sys.stdout = self._stdout
        sys.stderr = self._stderr
        self._console_file.close()
        self._console_file = None

    def _snapshot_params(self, params, exclude_keys=None):
        exclude_keys = exclude_keys or set()
        snapshot = {}
        for key, value in vars(params).items():
            if key in exclude_keys:
                continue
            snapshot[key] = normalize_for_json(value)
        return snapshot

    def _config_identity_payload(self, params):
        return self._snapshot_params(params, exclude_keys=CONFIG_ID_EXCLUDE_KEYS)

    def _environment_snapshot(self):
        return {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "hostname": socket.gethostname(),
            "git_commit": self._git_output(["rev-parse", "HEAD"]),
            "git_dirty": self._git_dirty(),
        }

    def _git_output(self, git_args):
        try:
            result = subprocess.run(
                ["git", *git_args],
                cwd=self.repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()
        except Exception:
            return None

    def _git_dirty(self):
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=self.repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
            return bool(result.stdout.strip())
        except Exception:
            return None

    def write_config_snapshot(self, params):
        payload = {
            "exp_id": self.exp_id,
            "run_id": self.run_id,
            "config_id": self.config_id,
            "argv": sys.argv,
            "resolved_params": self._snapshot_params(params, exclude_keys=SNAPSHOT_EXCLUDE_KEYS),
            "environment": self._environment_snapshot(),
            "paths": {
                "output_dir": self.suite_dir,
                "run_dir": self.run_dir,
            },
        }
        write_json(self.config_path, payload)

    def ensure_config_id(self, params):
        identity_payload = self._config_identity_payload(params)
        canonical_config = json.dumps(
            normalize_for_json(identity_payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        config_id = f"cfg_{hashlib.sha1(canonical_config.encode('utf-8')).hexdigest()[:12]}"

        self.canonical_config = json.loads(canonical_config)
        self.config_id = config_id
        self.status["config_id"] = config_id
        self.write_status()
        self.write_config_snapshot(params)
        self._register_config(params)
        return config_id

    def _register_config(self, params):
        timestamp = now_iso()
        record = {
            "config_id": self.config_id,
            "first_seen_at": timestamp,
            "last_seen_at": timestamp,
            "model": getattr(params, "model", None),
            "dataset": getattr(params, "dataset", None) or getattr(params, "downstream_dataset", None),
            "split_mode": getattr(params, "split_mode", None),
            "held_out_subject": getattr(params, "held_out_subject", None),
            "fold": getattr(params, "fold", None),
            "seed": getattr(params, "seed", None),
            "exp_ids": [self.exp_id],
            "canonical_config": self.canonical_config,
        }

        # Fast path: check if already registered without lock.
        # configs.jsonl is append-only; reading without lock is safe
        # except for the last line which may be partially written.
        if os.path.exists(self.configs_jsonl_path):
            try:
                with open(self.configs_jsonl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            payload = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if payload.get("config_id") == self.config_id:
                            exp_ids = list(payload.get("exp_ids") or [])
                            if self.exp_id in exp_ids:
                                # Already fully registered; nothing to do.
                                return
                            # Found but exp_id missing; fall through to locked update.
                            break
            except Exception:
                pass  # Fall through to locked path on any read error.

        with _gpfs_exclusive_lock(self.registry_lock_path):
            records = []
            found = False
            if os.path.exists(self.configs_jsonl_path):
                with open(self.configs_jsonl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            payload = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if payload.get("config_id") == self.config_id:
                            exp_ids = list(payload.get("exp_ids") or [])
                            if self.exp_id not in exp_ids:
                                exp_ids.append(self.exp_id)
                            payload["exp_ids"] = exp_ids
                            payload["last_seen_at"] = timestamp
                            found = True
                        records.append(payload)

            if not found:
                records.append(record)

            tmp_path = f"{self.configs_jsonl_path}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                for payload in records:
                    f.write(json.dumps(normalize_for_json(payload), ensure_ascii=False) + "\n")
            os.replace(tmp_path, self.configs_jsonl_path)

    def update_status(self, **kwargs):
        if "cache_usage" in kwargs:
            kwargs.update(resolve_status_fields_from_cache_usage(kwargs.get("cache_usage")))
        self.status.update(kwargs)
        if "config_id" not in kwargs:
            self.status["config_id"] = self.config_id
        self.write_status()

    def write_status(self):
        write_json(self.status_path, self.status)

    def mark_running(self):
        self.update_status(status="running", error_message=None)

    def mark_completed(self):
        finished_at = now_iso()
        started_at = self.status.get("started_at")
        duration_sec = None
        if started_at:
            duration_sec = int(
                (
                    datetime.fromisoformat(finished_at)
                    - datetime.fromisoformat(started_at)
                ).total_seconds()
            )

        self.update_status(
            status="completed",
            finished_at=finished_at,
            duration_sec=duration_sec,
            error_message=None,
        )

    def mark_failed(self, error_message):
        finished_at = now_iso()
        started_at = self.status.get("started_at")
        duration_sec = None
        if started_at:
            duration_sec = int(
                (
                    datetime.fromisoformat(finished_at)
                    - datetime.fromisoformat(started_at)
                ).total_seconds()
            )

        self.update_status(
            status="failed",
            finished_at=finished_at,
            duration_sec=duration_sec,
            error_message=error_message,
        )

    def record_failure(self, params, exc):
        """Best-effort bookkeeping for a failed run."""
        try:
            params.config_id = self.ensure_config_id(params)
        except Exception:
            params.config_id = self.config_id
        self.mark_failed(str(exc))
        self.append_experiment_row(params)

    def append_epoch_metrics(self, record):
        append_jsonl(self.epoch_metrics_path, record)

    def write_best_val_metrics(self, payload):
        write_json(self.best_val_metrics_path, payload)

    def write_final_test_metrics(self, payload):
        write_json(self.final_test_metrics_path, payload)

    def _ensure_experiments_csv_schema_locked(self):
        if not os.path.exists(self.experiments_csv_path) or os.path.getsize(self.experiments_csv_path) == 0:
            return

        with open(self.experiments_csv_path, "r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            current_headers = reader.fieldnames or []
            if current_headers == EXPERIMENTS_CSV_HEADERS:
                return
            existing_rows = list(reader)

        tmp_path = f"{self.experiments_csv_path}.tmp"
        with open(tmp_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=EXPERIMENTS_CSV_HEADERS)
            writer.writeheader()
            for row in existing_rows:
                writer.writerow({field: row.get(field) for field in EXPERIMENTS_CSV_HEADERS})
        os.replace(tmp_path, self.experiments_csv_path)

    def append_experiment_row(self, params):
        if self._experiment_row_written:
            return

        metadata_fields = resolve_run_metadata_fields(params)
        cache_link_fields = resolve_cache_link_fields(params)
        final_metric_fields = load_final_metric_fields(self.final_test_metrics_path)
        row = {
            "exp_id": self.exp_id,
            "run_id": self.run_id,
            "config_id": self.config_id,
            "data_config_id": metadata_fields["data_config_id"],
            "data_config_key": metadata_fields["data_config_key"],
            "suite_name": self.suite_name,
            "suite_dir": self.suite_dir,
            "run_dir": self.run_dir,
            "model": getattr(params, "model", None),
            "dataset": metadata_fields["dataset"],
            "task": metadata_fields["task"],
            "split_mode": getattr(params, "split_mode", None),
            "held_out_subject": getattr(params, "held_out_subject", None),
            "fold": getattr(params, "fold", None),
            "train_subjects": subject_list_text(self.status.get("train_subjects")),
            "val_subjects": subject_list_text(self.status.get("val_subjects")),
            "test_subjects": subject_list_text(self.status.get("test_subjects")),
            "seed": getattr(params, "seed", None),
            "channel_system": getattr(params, "channel_system", None),
            "channel_variant": getattr(params, "channel_variant", None),
            "channel_sparsity_config": getattr(params, "channel_sparsity_config", None),
            "channel_count_target": getattr(params, "channel_count_target", None),
            "channel_count_loaded": getattr(params, "channel_count_loaded", None),
            "input_format": getattr(params, "input_format", None),
            "variant": getattr(params, "variant", None),
            "epochs": getattr(params, "epochs", None),
            "batch_size": getattr(params, "batch_size", None),
            "data_ratio": getattr(params, "data_ratio", None),
            "train_samples_original": self.status.get("train_samples_original"),
            "train_samples": self.status.get("train_samples"),
            "val_samples_original": self.status.get("val_samples_original"),
            "val_samples": self.status.get("val_samples"),
            "test_samples_original": self.status.get("test_samples_original"),
            "test_samples": self.status.get("test_samples"),
            "lr": getattr(params, "lr", None),
            "encoder_lr_scale": getattr(params, "encoder_lr_scale", None),
            "classifier_lr_scale": getattr(params, "classifier_lr_scale", None),
            "epoch_method": metadata_fields["epoch_method"],
            "pipeline_key": metadata_fields["pipeline_key"],
            "pipeline": metadata_fields["pipeline"],
            "pipeline_params": metadata_fields["pipeline_params"],
            "epoch_params": metadata_fields["epoch_params"],
            "cache_root": cache_link_fields["cache_root"],
            "locator_cache_id": cache_link_fields["locator_cache_id"],
            "hdf5_cache_id": cache_link_fields["hdf5_cache_id"],
            "status": self.status.get("status"),
            "best_epoch": self.status.get("best_epoch"),
            "selection_metric_name": self.status.get("selection_metric_name"),
            "selection_metric_value": self.status.get("selection_metric_value"),
            **final_metric_fields,
            "started_at": self.status.get("started_at"),
            "finished_at": self.status.get("finished_at"),
            "duration_sec": self.status.get("duration_sec"),
            "config_path": self.config_path,
            "status_path": self.status_path,
            "final_test_metrics_path": self.status.get("final_test_metrics_path"),
            "log_path": self.console_log_path,
        }

        # Fast path: check if row already exists without lock.
        if os.path.exists(self.experiments_csv_path) and os.path.getsize(self.experiments_csv_path) > 0:
            try:
                with open(self.experiments_csv_path, "r", newline="", encoding="utf-8") as csv_file:
                    reader = csv.DictReader(csv_file)
                    for existing_row in reader:
                        if existing_row.get("run_id") == self.run_id:
                            self._experiment_row_written = True
                            return
            except Exception:
                pass  # Fall through to locked path on any read error.

        with _gpfs_exclusive_lock(self.registry_lock_path):
            # Double-check after acquiring lock.
            if os.path.exists(self.experiments_csv_path) and os.path.getsize(self.experiments_csv_path) > 0:
                try:
                    with open(self.experiments_csv_path, "r", newline="", encoding="utf-8") as csv_file:
                        reader = csv.DictReader(csv_file)
                        for existing_row in reader:
                            if existing_row.get("run_id") == self.run_id:
                                self._experiment_row_written = True
                                return
                except Exception:
                    pass

            self._ensure_experiments_csv_schema_locked()
            file_exists = os.path.exists(self.experiments_csv_path)
            with open(self.experiments_csv_path, "a", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=EXPERIMENTS_CSV_HEADERS)
                if not file_exists or os.path.getsize(self.experiments_csv_path) == 0:
                    writer.writeheader()
                writer.writerow(normalize_for_json(row))

        refresh_registry_views(self.layout, self.repo_root)
        self._experiment_row_written = True
