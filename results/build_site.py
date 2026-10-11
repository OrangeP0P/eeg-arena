from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import sys
import tempfile
from collections import Counter, defaultdict
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass, field
from datetime import datetime
from html import escape
from io import StringIO
from pathlib import Path
from typing import Any

from channels import PUBLIC_CHANNEL_SYSTEM_CHOICES

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "output"

PRIMARY_METRIC_FALLBACK = ("kappa", "f1_macro", "f1_weighted", "acc")

STATUS_MAP = {
    "Success": "completed",
    "Failed": "failed",
    "Running": "running",
    "Pending": "pending",
    "completed": "completed",
    "failed": "failed",
    "running": "running",
    "pending": "pending",
}
STATUS_BUCKET_PRIORITY = {
    "completed": 5,
    "artifacts_missing": 4,
    "failed": 3,
    "running": 2,
    "pending": 1,
}

MODEL_FAMILY_MAP = {
    "atcnet": "raw",
    "bendr": "raw",
    "biot": "raw",
    "brainomni": "foundation",
    "cbramod": "patched",
    "codebrain": "foundation",
    "csbrain": "patched",
    "deepconvnet": "raw",
    "eegconformer": "raw",
    "eeginception_erp": "raw",
    "eegmamba": "raw",
    "eegnet": "raw",
    "eegnex": "raw",
    "eegpt": "patched",
    "femba": "foundation",
    "labram": "patched",
    "lead": "foundation",
    "luna": "foundation",
    "mdjpt": "foundation",
    "mirepnet": "foundation",
    "neurogpt": "patched",
    "neurolm": "foundation",
    "neurorvq": "foundation",
    "reve": "raw",
    "shallowconvnet": "raw",
    "steegformer": "raw",
    "singlem": "foundation",
}

BASE_FIELD_ORDER = [
    "effective_id",
    "raw_id",
    "condition_id",
    "is_selected_effective",
    "suite_name",
    "suite_label",
    "suite_dir",
    "job_name",
    "task_index",
    "status_bucket",
    "status",
    "model",
    "model_family",
    "dataset",
    "task",
    "dataset_key",
    "split_mode",
    "fold",
    "held_out_subject",
    "seed",
    "channel_system",
    "channel_sparsity_config",
    "input_format",
    "variant",
    "epochs",
    "batch_size",
    "lr",
    "encoder_lr_scale",
    "classifier_lr_scale",
    "epoch_method",
    "pipeline",
    "pipeline_key",
    "pipeline_params_json",
    "epoch_params_json",
    "selection_metric_name",
    "selection_metric_value",
    "primary_metric_name",
    "primary_metric_value",
    "final_acc",
    "final_f1_macro",
    "final_f1_weighted",
    "final_kappa",
    "best_epoch",
    "started_at",
    "finished_at",
    "duration_sec",
    "exp_id",
    "run_id",
    "config_id",
    "run_dir",
    "config_path",
    "status_path",
    "best_val_metrics_path",
    "final_test_metrics_path",
    "epoch_metrics_path",
    "log_path",
]

IDENTITY_BASE_KEYS = {
    "model",
    "dataset",
    "task",
    "split_mode",
    "fold",
    "held_out_subject",
    "seed",
}
IDENTITY_EXACT_EXCLUDES = {
    "argv",
    "batch_sampler",
    "best_checkpoint_path",
    "best_val_metrics_path",
    "cache_root",
    "cache_dir",
    "checkpoint_dir",
    "ch_names",
    "ch_positions",
    "class_mapping",
    "config_id",
    "config_path",
    "console_log_path",
    "cuda",
    "data_config_id",
    "data_config_key",
    "dataset",
    "device",
    "device_obj",
    "duration_sec",
    "epoch_metrics_path",
    "exp_id",
    "final_test_metrics_path",
    "finished_at",
    "foundation_dir",
    "gpu",
    "gpu_ids",
    "hdf5_cache_id",
    "held_out_subject",
    "job_name",
    "label_mapping",
    "last_checkpoint_path",
    "locator_cache_id",
    "log_path",
    "num_of_classes",
    "output_dir",
    "paths",
    "pid",
    "python_executable",
    "return_code",
    "run_dir",
    "run_id",
    "seed",
    "started_at",
    "status_path",
    "suite_dir",
    "suite_name",
    "task",
    "task_index",
    "test_subjects",
    "train_subjects",
    "use_dataparallel",
    "val_subjects",
}
IDENTITY_PREFIX_EXCLUDES = (
    "cache_usage.",
    "environment.",
    "path.",
    "paths.",
    "runtime.",
)

DISCOVERY_ENV_VARS = {
    "MNE_DONTWRITE_HOME": "true",
}

CORE_FILTER_KEYS = {
    "suite_name",
    "model",
    "selection_metric_name",
    "split_mode",
    "dataset_key",
    "channel_system",
    "input_format",
    "classifier_lr_scale",
    "status_bucket",
}

CORE_FILTER_PRIORITY = [
    "suite_name",
    "model",
    "selection_metric_name",
    "split_mode",
    "dataset_key",
    "channel_system",
    "input_format",
    "classifier_lr_scale",
    "status_bucket",
]

FILTER_GROUP_OVERRIDES = {
    "suite_name": "Core",
    "status_bucket": "Core",
    "dataset_key": "Core",
    "dataset": "Core",
    "task": "Core",
    "model": "Core",
    "split_mode": "Core",
    "channel_system": "Core",
    "classifier_lr_scale": "Core",
    "input_format": "Core",
    "variant": "Model Scoped",
    "selection_metric_name": "Core",
    "pipeline_params": "Structured",
    "epoch_params": "Structured",
    "resolved_params": "Resolved",
}

STATUS_LABELS = {
    "completed": "Completed",
    "artifacts_missing": "Artifacts Missing",
    "failed": "Failed",
    "running": "Running",
    "pending": "Pending",
}


@dataclass
class RunRecord:
    raw_id: str
    suite_name: str
    suite_label: str
    suite_dir: str
    suite_index: int
    task_index: str
    job_name: str
    status: str
    status_bucket: str
    is_rankable: bool
    exp_id: str
    run_id: str
    config_id: str
    model: str
    model_family: str
    dataset: str
    task: str
    dataset_key: str
    split_mode: str
    fold: str
    held_out_subject: str
    seed: str
    channel_system: str
    channel_sparsity_config: str
    input_format: str
    variant: str
    epochs: str
    batch_size: str
    lr: str
    encoder_lr_scale: str
    classifier_lr_scale: str
    epoch_method: str
    pipeline: str
    pipeline_key: str
    pipeline_params_json: str
    epoch_params_json: str
    selection_metric_name: str
    selection_metric_value: float | None
    final_acc: float | None
    final_f1_macro: float | None
    final_f1_weighted: float | None
    final_kappa: float | None
    best_epoch: int | None
    started_at: str
    finished_at: str
    duration_sec: float | None
    train_subject_count: int | None
    val_subject_count: int | None
    test_subject_count: int | None
    run_dir: str
    config_path: str
    status_path: str
    best_val_metrics_path: str
    final_test_metrics_path: str
    epoch_metrics_path: str
    log_path: str
    condition_id: str = ""
    primary_metric_name: str = ""
    primary_metric_value: float | None = None
    confusion_matrix: list[list[float]] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    raw_payload: dict[str, Any] = field(default_factory=dict)

    def to_payload_dict(self) -> dict[str, Any]:
        row = {
            "raw_id": self.raw_id,
            "suite_name": self.suite_name,
            "suite_label": self.suite_label,
            "suite_dir": self.suite_dir,
            "suite_index": self.suite_index,
            "task_index": self.task_index,
            "job_name": self.job_name,
            "status": self.status,
            "status_bucket": self.status_bucket,
            "is_rankable": self.is_rankable,
            "exp_id": self.exp_id,
            "run_id": self.run_id,
            "config_id": self.config_id,
            "model": self.model,
            "model_family": self.model_family,
            "dataset": self.dataset,
            "task": self.task,
            "dataset_key": self.dataset_key,
            "split_mode": self.split_mode,
            "fold": self.fold,
            "held_out_subject": self.held_out_subject,
            "seed": self.seed,
            "channel_system": self.channel_system,
            "channel_sparsity_config": self.channel_sparsity_config,
            "input_format": self.input_format,
            "variant": self.variant,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "encoder_lr_scale": self.encoder_lr_scale,
            "classifier_lr_scale": self.classifier_lr_scale,
            "epoch_method": self.epoch_method,
            "pipeline": self.pipeline,
            "pipeline_key": self.pipeline_key,
            "pipeline_params_json": self.pipeline_params_json,
            "epoch_params_json": self.epoch_params_json,
            "selection_metric_name": self.selection_metric_name,
            "selection_metric_value": self.selection_metric_value,
            "primary_metric_name": self.primary_metric_name,
            "primary_metric_value": self.primary_metric_value,
            "final_acc": self.final_acc,
            "final_f1_macro": self.final_f1_macro,
            "final_f1_weighted": self.final_f1_weighted,
            "final_kappa": self.final_kappa,
            "best_epoch": self.best_epoch,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_sec": self.duration_sec,
            "train_subject_count": self.train_subject_count,
            "val_subject_count": self.val_subject_count,
            "test_subject_count": self.test_subject_count,
            "run_dir": self.run_dir,
            "config_path": self.config_path,
            "status_path": self.status_path,
            "best_val_metrics_path": self.best_val_metrics_path,
            "final_test_metrics_path": self.final_test_metrics_path,
            "epoch_metrics_path": self.epoch_metrics_path,
            "log_path": self.log_path,
            "condition_id": self.condition_id,
            "confusion_matrix": self.confusion_matrix,
            "params_json": stable_json_text(self.params),
        }
        row.update({key: canonical_text(value) for key, value in self.params.items()})
        return row


@dataclass
class EffectiveRecord:
    effective_id: str
    condition_id: str
    selected_raw_id: str
    selected_suite_name: str
    status_bucket: str
    is_rankable: bool
    primary_metric_name: str
    primary_metric_value: float | None
    source_history: list[dict[str, Any]]
    curve_series: list[dict[str, Any]] = field(default_factory=list)
    source_count: int = 0
    selected_record: dict[str, Any] = field(default_factory=dict)

    def to_payload_dict(self) -> dict[str, Any]:
        row = dict(self.selected_record)
        row["effective_id"] = self.effective_id
        row["condition_id"] = self.condition_id
        row["selected_raw_id"] = self.selected_raw_id
        row["selected_suite_name"] = self.selected_suite_name
        row["status_bucket"] = self.status_bucket
        row["is_rankable"] = self.is_rankable
        row["primary_metric_name"] = self.primary_metric_name
        row["primary_metric_value"] = self.primary_metric_value
        row["source_history"] = self.source_history
        row["source_count"] = self.source_count
        row["curve_series"] = self.curve_series
        row["source_suite_names"] = ", ".join(item["suite_name"] for item in self.source_history)
        row["source_statuses"] = ", ".join(item["status_bucket"] for item in self.source_history)
        row["source_history_json"] = stable_json_text(self.source_history)
        row["curve_series_json"] = stable_json_text(self.curve_series)
        return row


@dataclass
class FilterField:
    key: str
    label: str
    group: str
    kind: str
    help_text: str
    default_value: str
    observed_values: list[str]
    choice_values: list[str]
    render_values: list[str]
    core: bool
    enabled: bool
    disabled_reason: str
    models: list[str] = field(default_factory=list)


@dataclass
class SitePayload:
    site_name: str
    generated_at: str
    summary: dict[str, Any]
    suites: list[dict[str, Any]]
    filters: list[dict[str, Any]]
    dataset_primary_metrics: dict[str, str]
    model_order: list[str]
    dataset_order: list[str]
    raw_runs: list[dict[str, Any]]
    effective_runs: list[dict[str, Any]]


@dataclass
class SuiteSource:
    index: int
    name: str
    label: str
    path: Path
    manifest: dict[str, Any]
    summary: dict[str, Any]
    state: dict[str, Any]
    task_rows: list[dict[str, str]]
    manifest_by_job: dict[str, dict[str, Any]]

    def metadata(self) -> dict[str, Any]:
        return {
            "suite_name": self.name,
            "suite_label": self.label,
            "suite_dir": str(self.path),
            "config_path": canonical_text(self.manifest.get("suite_config_path")),
            "total_tasks": parse_int(self.summary.get("total_tasks")) or len(self.task_rows),
            "completed_count": parse_int(self.summary.get("completed_count")) or 0,
            "failed_count": parse_int(self.summary.get("failed_count")) or 0,
            "models": list(self.manifest.get("models") or []),
            "datasets": list(self.manifest.get("datasets") or []),
            "sweeps": self.manifest.get("sweeps") or {},
            "progress": self.state.get("progress") or {},
            "finished_at": canonical_text(self.summary.get("finished_at")),
        }


def canonical_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        if value.is_integer():
            return str(int(value))
        return format(value, ".12g")
    if isinstance(value, int):
        return str(value)
    return str(value).strip()


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    text = canonical_text(value)
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def parse_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return int(value)
    text = canonical_text(value)
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def parse_time(value: Any) -> datetime | None:
    text = canonical_text(value)
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def stable_json_text(value: Any) -> str:
    if value in (None, "", [], {}):
        return ""
    return json.dumps(json_ready(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                item = json.loads(text)
                if isinstance(item, dict):
                    rows.append(item)
    except (OSError, json.JSONDecodeError):
        return []
    return rows


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def slugify(value: str) -> str:
    cleaned = []
    for char in value.lower():
        if char.isalnum():
            cleaned.append(char)
        elif cleaned and cleaned[-1] != "_":
            cleaned.append("_")
    return "".join(cleaned).strip("_") or "results_dashboard"


def parse_json_text(value: Any) -> Any:
    text = canonical_text(value)
    if not text or text[0] not in "[{":
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def parse_argv_flags(argv: Any) -> dict[str, Any]:
    if not isinstance(argv, list):
        return {}
    result: dict[str, Any] = {}
    index = 0
    while index < len(argv):
        token = canonical_text(argv[index])
        if not token.startswith("--"):
            index += 1
            continue
        key = token[2:].replace("-", "_")
        next_token = canonical_text(argv[index + 1]) if index + 1 < len(argv) else ""
        if next_token and not next_token.startswith("--"):
            result[key] = next_token
            index += 2
        else:
            result[key] = "true"
            index += 1
    return result


def metrics_block(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload.get("metrics")
    return metrics if isinstance(metrics, dict) else payload


def extract_metrics(payload: dict[str, Any]) -> tuple[dict[str, float | None], list[list[float]]]:
    block = metrics_block(payload)
    confusion = block.get("confusion_matrix")
    matrix: list[list[float]] = []
    if isinstance(confusion, list):
        for row in confusion:
            if not isinstance(row, list):
                continue
            matrix.append([parse_float(item) or 0.0 for item in row])
    return (
        {
            "acc": parse_float(block.get("acc")),
            "f1_macro": parse_float(block.get("f1_macro")),
            "f1_weighted": parse_float(block.get("f1_weighted")),
            "kappa": parse_float(block.get("kappa")),
        },
        matrix,
    )


def subject_count(value: Any) -> int | None:
    parsed = parse_json_text(value)
    if isinstance(parsed, list):
        return len(parsed)
    text = canonical_text(value)
    if not text:
        return None
    return None


def pick_first(*values: Any) -> Any:
    for value in values:
        text = canonical_text(value)
        if text:
            return value
    return None


def normalize_status_bucket(status_value: Any, has_final_metrics: bool, has_status: bool) -> str:
    normalized = STATUS_MAP.get(canonical_text(status_value), "")
    if normalized == "completed" and not (has_final_metrics and has_status):
        return "artifacts_missing"
    if normalized:
        return normalized
    return "pending"


def model_family(model: str, input_format: str = "") -> str:
    family = MODEL_FAMILY_MAP.get(model)
    if family:
        return family
    if input_format == "patched":
        return "patched"
    if input_format == "raw":
        return "raw"
    return "unknown"


def path_or_empty(path: Path | None) -> str:
    if path is None:
        return ""
    return str(path)


def load_suite_sources(suite_dirs: list[Path]) -> list[SuiteSource]:
    base_names = [path.name for path in suite_dirs]
    counts = Counter(base_names)
    used_counts: Counter[str] = Counter()
    suites: list[SuiteSource] = []

    for index, suite_dir in enumerate(suite_dirs):
        scheduler_dir = suite_dir / "scheduler"
        task_status_path = scheduler_dir / "task_status.csv"
        if not task_status_path.exists():
            raise FileNotFoundError(f"Missing task_status.csv under {scheduler_dir}")
        task_manifest_path = scheduler_dir / "task_manifest.json"
        summary_path = scheduler_dir / "summary.json"
        state_path = scheduler_dir / "state.json"

        label = suite_dir.name
        if counts[label] > 1:
            used_counts[label] += 1
            name = f"{label}__{used_counts[label]}"
        else:
            name = label

        manifest = read_json(task_manifest_path)
        summary = read_json(summary_path)
        state = read_json(state_path)
        task_rows = read_csv_rows(task_status_path)
        manifest_by_job = {}
        for item in manifest.get("tasks", []):
            if isinstance(item, dict) and canonical_text(item.get("job_name")):
                manifest_by_job[canonical_text(item.get("job_name"))] = item

        suites.append(
            SuiteSource(
                index=index,
                name=name,
                label=label,
                path=suite_dir,
                manifest=manifest,
                summary=summary,
                state=state,
                task_rows=task_rows,
                manifest_by_job=manifest_by_job,
            )
        )
    return suites


def should_keep_identity_key(key: str, value: Any) -> bool:
    if key in IDENTITY_BASE_KEYS:
        return False
    if key in IDENTITY_EXACT_EXCLUDES:
        return False
    if any(key.startswith(prefix) for prefix in IDENTITY_PREFIX_EXCLUDES):
        return False
    if value is None:
        return False
    if isinstance(value, (dict, list, tuple)):
        return False
    text = canonical_text(value)
    return bool(text)


def should_capture_param(key: str, value: Any) -> bool:
    if value is None:
        return False
    if key in {
        "argv",
        "environment",
        "paths",
        "cache_usage",
        "train_subjects",
        "val_subjects",
        "test_subjects",
        "ch_names",
        "ch_positions",
        "label_mapping",
        "class_mapping",
    }:
        return False
    if isinstance(value, (dict, list, tuple)):
        return False
    return bool(canonical_text(value))


def flatten_scalar_params(params: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in params.items():
        if key in {"pipeline_params", "epoch_params"}:
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if should_capture_param(sub_key, sub_value):
                        flattened[f"{key}.{sub_key}"] = sub_value
            continue
        if should_capture_param(key, value):
            flattened[key] = value
    return flattened


def build_condition_id(record: RunRecord) -> str:
    identity_payload: dict[str, Any] = {
        "dataset": record.dataset,
        "task": record.task,
        "split_mode": record.split_mode,
        "fold": record.fold,
        "held_out_subject": record.held_out_subject,
        "seed": record.seed,
        "params": {},
    }
    for key in sorted(record.params):
        value = record.params[key]
        if should_keep_identity_key(key, value):
            identity_payload["params"][key] = canonical_text(value)
    digest = hashlib.sha1(
        json.dumps(identity_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return f"cond_{digest}"


def select_task_value(task_row: dict[str, str], status_payload: dict[str, Any], key: str) -> Any:
    return pick_first(task_row.get(key), status_payload.get(key))


def build_run_records(suites: list[SuiteSource]) -> list[RunRecord]:
    records: list[RunRecord] = []
    for suite in suites:
        for task_row in suite.task_rows:
            job_name = canonical_text(task_row.get("job_name"))
            manifest_task = suite.manifest_by_job.get(job_name, {})
            manifest_args = manifest_task.get("run_args") if isinstance(manifest_task.get("run_args"), dict) else {}

            run_relpath = canonical_text(task_row.get("run_relpath"))
            if not run_relpath and canonical_text(task_row.get("run_id")):
                run_relpath = f"runs/{canonical_text(task_row.get('run_id'))}"

            run_dir = suite.path / run_relpath if run_relpath else Path()
            config_path = suite.path / canonical_text(task_row.get("config_relpath")) if canonical_text(task_row.get("config_relpath")) else (run_dir / "config.json" if run_relpath else Path())
            status_path = suite.path / canonical_text(task_row.get("status_relpath")) if canonical_text(task_row.get("status_relpath")) else (run_dir / "status.json" if run_relpath else Path())
            final_test_metrics_path = suite.path / canonical_text(task_row.get("final_test_metrics_relpath")) if canonical_text(task_row.get("final_test_metrics_relpath")) else (run_dir / "metrics" / "final_test_metrics.json" if run_relpath else Path())
            log_path = suite.path / canonical_text(task_row.get("log_relpath")) if canonical_text(task_row.get("log_relpath")) else (run_dir / "logs" / "console.log" if run_relpath else Path())
            best_val_metrics_path = run_dir / "metrics" / "best_val_metrics.json" if run_relpath else Path()
            epoch_metrics_path = run_dir / "metrics" / "epoch_metrics.jsonl" if run_relpath else Path()

            config_payload = read_json(config_path) if config_path.exists() else {}
            status_payload = read_json(status_path) if status_path.exists() else {}
            final_payload = read_json(final_test_metrics_path) if final_test_metrics_path.exists() else {}
            best_val_payload = read_json(best_val_metrics_path) if best_val_metrics_path.exists() else {}
            resolved_params = config_payload.get("resolved_params") if isinstance(config_payload.get("resolved_params"), dict) else {}
            argv_flags = parse_argv_flags(config_payload.get("argv"))
            cache_usage = status_payload.get("cache_usage") if isinstance(status_payload.get("cache_usage"), dict) else {}

            pipeline_params = (
                parse_json_text(task_row.get("pipeline_params"))
                if isinstance(parse_json_text(task_row.get("pipeline_params")), dict)
                else cache_usage.get("pipeline_params")
                if isinstance(cache_usage.get("pipeline_params"), dict)
                else manifest_args.get("pipeline_params")
                if isinstance(manifest_args.get("pipeline_params"), dict)
                else {}
            )
            epoch_params = (
                parse_json_text(task_row.get("epoch_params"))
                if isinstance(parse_json_text(task_row.get("epoch_params")), dict)
                else cache_usage.get("epoch_params")
                if isinstance(cache_usage.get("epoch_params"), dict)
                else manifest_args.get("epoch_params")
                if isinstance(manifest_args.get("epoch_params"), dict)
                else {}
            )

            metrics, confusion_matrix = extract_metrics(final_payload)
            best_epoch = parse_int(pick_first(task_row.get("best_epoch"), status_payload.get("best_epoch"), final_payload.get("epoch")))

            merged_params: dict[str, Any] = {}
            for layer in (manifest_args, argv_flags, resolved_params):
                for key, value in flatten_scalar_params(layer).items():
                    merged_params[key] = value
            if pipeline_params:
                for key, value in pipeline_params.items():
                    merged_params[f"pipeline_params.{key}"] = value
            if epoch_params:
                for key, value in epoch_params.items():
                    merged_params[f"epoch_params.{key}"] = value

            merged_params.setdefault("model", pick_first(task_row.get("model"), manifest_task.get("model")))
            merged_params.setdefault("dataset", pick_first(task_row.get("dataset"), status_payload.get("dataset")))
            merged_params.setdefault("task", pick_first(task_row.get("task"), status_payload.get("task")))
            merged_params.setdefault("split_mode", pick_first(task_row.get("split_mode"), manifest_args.get("split_mode"), status_payload.get("split_mode")))
            merged_params.setdefault("fold", pick_first(task_row.get("fold"), manifest_args.get("fold"), status_payload.get("fold")))
            merged_params.setdefault("held_out_subject", pick_first(task_row.get("held_out_subject"), manifest_args.get("held_out_subject"), status_payload.get("held_out_subject")))
            merged_params.setdefault("seed", pick_first(task_row.get("seed"), manifest_args.get("seed"), status_payload.get("seed")))
            merged_params.setdefault("channel_system", pick_first(task_row.get("channel_system"), manifest_args.get("channel_system"), resolved_params.get("channel_system")))
            merged_params.setdefault("channel_sparsity_config", pick_first(task_row.get("channel_sparsity_config"), manifest_args.get("channel_sparsity_config"), status_payload.get("channel_sparsity_config"), resolved_params.get("channel_sparsity_config")))
            merged_params.setdefault("input_format", pick_first(task_row.get("input_format"), manifest_args.get("input_format"), resolved_params.get("input_format")))
            merged_params.setdefault("variant", pick_first(task_row.get("variant"), manifest_args.get("variant"), resolved_params.get("variant")))
            merged_params.setdefault("epochs", pick_first(task_row.get("epochs"), manifest_args.get("epochs"), resolved_params.get("epochs")))
            merged_params.setdefault("batch_size", pick_first(task_row.get("batch_size"), manifest_args.get("batch_size"), resolved_params.get("batch_size")))
            merged_params.setdefault("lr", pick_first(manifest_args.get("lr"), resolved_params.get("lr")))
            merged_params.setdefault("encoder_lr_scale", pick_first(manifest_args.get("encoder_lr_scale"), resolved_params.get("encoder_lr_scale")))
            merged_params.setdefault("classifier_lr_scale", pick_first(task_row.get("classifier_lr_scale"), manifest_args.get("classifier_lr_scale"), resolved_params.get("classifier_lr_scale")))
            merged_params.setdefault("epoch_method", pick_first(task_row.get("epoch_method"), status_payload.get("epoch_method"), cache_usage.get("epoch_method")))
            merged_params.setdefault("pipeline", pick_first(task_row.get("pipeline"), manifest_args.get("pipeline"), resolved_params.get("pipeline"), cache_usage.get("pipeline")))
            merged_params.setdefault("pipeline_key", pick_first(task_row.get("pipeline_key"), status_payload.get("pipeline_key"), cache_usage.get("pipeline_key")))

            status_bucket = normalize_status_bucket(task_row.get("status") or status_payload.get("status"), final_test_metrics_path.exists(), status_path.exists())
            selection_metric_name = canonical_text(
                pick_first(
                    task_row.get("selection_metric_name"),
                    status_payload.get("selection_metric_name"),
                    final_payload.get("selection_metric_name"),
                )
            )
            selection_metric_value = parse_float(
                pick_first(
                    task_row.get("selection_metric_value"),
                    status_payload.get("selection_metric_value"),
                    final_payload.get("selection_metric_value"),
                )
            )

            record = RunRecord(
                raw_id=f"{suite.name}:{job_name or canonical_text(task_row.get('task_index'))}",
                suite_name=suite.name,
                suite_label=suite.label,
                suite_dir=str(suite.path),
                suite_index=suite.index,
                task_index=canonical_text(task_row.get("task_index")),
                job_name=job_name,
                status=canonical_text(task_row.get("status")) or canonical_text(status_payload.get("status")) or "Pending",
                status_bucket=status_bucket,
                is_rankable=status_bucket == "completed" and any(value is not None for value in metrics.values()),
                exp_id=canonical_text(pick_first(task_row.get("exp_id"), config_payload.get("exp_id"), status_payload.get("exp_id"))),
                run_id=canonical_text(pick_first(task_row.get("run_id"), status_payload.get("run_id"))),
                config_id=canonical_text(pick_first(task_row.get("config_id"), config_payload.get("config_id"), status_payload.get("config_id"))),
                model=canonical_text(merged_params.get("model")),
                model_family=model_family(canonical_text(merged_params.get("model")), canonical_text(merged_params.get("input_format"))),
                dataset=canonical_text(merged_params.get("dataset")),
                task=canonical_text(merged_params.get("task")),
                dataset_key=f"{canonical_text(merged_params.get('dataset'))}:{canonical_text(merged_params.get('task'))}",
                split_mode=canonical_text(merged_params.get("split_mode")),
                fold=canonical_text(merged_params.get("fold")),
                held_out_subject=canonical_text(merged_params.get("held_out_subject")),
                seed=canonical_text(merged_params.get("seed")),
                channel_system=canonical_text(merged_params.get("channel_system")),
                channel_sparsity_config=canonical_text(merged_params.get("channel_sparsity_config")),
                input_format=canonical_text(merged_params.get("input_format")),
                variant=canonical_text(merged_params.get("variant")),
                epochs=canonical_text(merged_params.get("epochs")),
                batch_size=canonical_text(merged_params.get("batch_size")),
                lr=canonical_text(merged_params.get("lr")),
                encoder_lr_scale=canonical_text(merged_params.get("encoder_lr_scale")),
                classifier_lr_scale=canonical_text(merged_params.get("classifier_lr_scale")),
                epoch_method=canonical_text(merged_params.get("epoch_method")),
                pipeline=canonical_text(merged_params.get("pipeline")),
                pipeline_key=canonical_text(merged_params.get("pipeline_key")),
                pipeline_params_json=stable_json_text(pipeline_params),
                epoch_params_json=stable_json_text(epoch_params),
                selection_metric_name=selection_metric_name,
                selection_metric_value=selection_metric_value,
                final_acc=metrics["acc"],
                final_f1_macro=metrics["f1_macro"],
                final_f1_weighted=metrics["f1_weighted"],
                final_kappa=metrics["kappa"],
                best_epoch=best_epoch,
                started_at=canonical_text(pick_first(task_row.get("started_at"), status_payload.get("started_at"))),
                finished_at=canonical_text(pick_first(task_row.get("finished_at"), status_payload.get("finished_at"), suite.summary.get("finished_at"))),
                duration_sec=parse_float(pick_first(task_row.get("duration_sec"), status_payload.get("duration_sec"))),
                train_subject_count=subject_count(pick_first(task_row.get("train_subjects"), status_payload.get("train_subjects"))),
                val_subject_count=subject_count(pick_first(task_row.get("val_subjects"), status_payload.get("val_subjects"))),
                test_subject_count=subject_count(pick_first(task_row.get("test_subjects"), status_payload.get("test_subjects"))),
                run_dir=path_or_empty(run_dir if run_relpath else None),
                config_path=path_or_empty(config_path if config_path.exists() else None),
                status_path=path_or_empty(status_path if status_path.exists() else None),
                best_val_metrics_path=path_or_empty(best_val_metrics_path if best_val_metrics_path.exists() else None),
                final_test_metrics_path=path_or_empty(final_test_metrics_path if final_test_metrics_path.exists() else None),
                epoch_metrics_path=path_or_empty(epoch_metrics_path if epoch_metrics_path.exists() else None),
                log_path=path_or_empty(log_path if log_path.exists() else None),
                confusion_matrix=confusion_matrix,
                params=merged_params,
                raw_payload={
                    "task_status": task_row,
                    "manifest_task": manifest_task,
                    "config": config_payload,
                    "status": status_payload,
                    "best_val": best_val_payload,
                    "final_test": final_payload,
                },
            )
            record.condition_id = build_condition_id(record)
            records.append(record)
    return records


def derive_primary_metrics(raw_records: list[RunRecord]) -> dict[str, str]:
    selection_counts: dict[str, Counter[str]] = defaultdict(Counter)
    availability: dict[str, Counter[str]] = defaultdict(Counter)
    all_dataset_keys = {record.dataset_key for record in raw_records}

    for record in raw_records:
        if record.status_bucket != "completed":
            continue
        if record.selection_metric_name:
            selection_counts[record.dataset_key][record.selection_metric_name] += 1
        if record.final_acc is not None:
            availability[record.dataset_key]["acc"] += 1
        if record.final_f1_macro is not None:
            availability[record.dataset_key]["f1_macro"] += 1
        if record.final_f1_weighted is not None:
            availability[record.dataset_key]["f1_weighted"] += 1
        if record.final_kappa is not None:
            availability[record.dataset_key]["kappa"] += 1

    result: dict[str, str] = {}
    for dataset_key in sorted(all_dataset_keys):
        if selection_counts[dataset_key]:
            metric = sorted(
                selection_counts[dataset_key].items(),
                key=lambda item: (-item[1], PRIMARY_METRIC_FALLBACK.index(item[0]) if item[0] in PRIMARY_METRIC_FALLBACK else 999, item[0]),
            )[0][0]
            result[dataset_key] = metric
            continue
        metric = next((name for name in PRIMARY_METRIC_FALLBACK if availability[dataset_key][name] > 0), "acc")
        result[dataset_key] = metric
    return result


def attach_primary_metrics(raw_records: list[RunRecord], dataset_primary_metrics: dict[str, str]) -> None:
    for record in raw_records:
        metric_name = dataset_primary_metrics.get(record.dataset_key, "acc")
        record.primary_metric_name = metric_name
        if metric_name == "acc":
            record.primary_metric_value = record.final_acc
        elif metric_name == "f1_macro":
            record.primary_metric_value = record.final_f1_macro
        elif metric_name == "f1_weighted":
            record.primary_metric_value = record.final_f1_weighted
        else:
            record.primary_metric_value = record.final_kappa
        record.is_rankable = record.status_bucket == "completed" and record.primary_metric_value is not None


def record_selection_key(record: RunRecord) -> tuple[int, float, float, int]:
    finished = parse_time(record.finished_at)
    started = parse_time(record.started_at)
    return (
        STATUS_BUCKET_PRIORITY.get(record.status_bucket, 0),
        finished.timestamp() if finished else float("-inf"),
        started.timestamp() if started else float("-inf"),
        record.suite_index,
    )


def source_history_entry(record: RunRecord, selected_raw_id: str) -> dict[str, Any]:
    return {
        "raw_id": record.raw_id,
        "suite_name": record.suite_name,
        "status": record.status,
        "status_bucket": record.status_bucket,
        "run_id": record.run_id,
        "exp_id": record.exp_id,
        "selection_metric_name": record.selection_metric_name,
        "selection_metric_value": record.selection_metric_value,
        "primary_metric_name": record.primary_metric_name,
        "primary_metric_value": record.primary_metric_value,
        "finished_at": record.finished_at,
        "started_at": record.started_at,
        "duration_sec": record.duration_sec,
        "is_selected": record.raw_id == selected_raw_id,
    }


def load_curve_series(record: RunRecord) -> list[dict[str, Any]]:
    path = Path(record.epoch_metrics_path)
    if not record.is_rankable or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for item in read_jsonl(path):
        epoch = parse_int(item.get("epoch"))
        if epoch is None:
            continue
        val_metrics = item.get("val_metrics") if isinstance(item.get("val_metrics"), dict) else {}
        test_metrics = item.get("test_metrics") if isinstance(item.get("test_metrics"), dict) else {}
        lr_groups = item.get("lr_groups") if isinstance(item.get("lr_groups"), list) else []
        primary_key = record.primary_metric_name or "acc"
        rows.append(
            {
                "epoch": epoch,
                "is_best": bool(item.get("is_best")),
                "val_primary": parse_float(val_metrics.get(primary_key)),
                "test_primary": parse_float(test_metrics.get(primary_key)),
                "train_loss": parse_float(item.get("train_loss")),
                "lr": parse_float(lr_groups[0]) if lr_groups else None,
            }
        )
    return rows


def build_effective_records(raw_records: list[RunRecord]) -> tuple[list[EffectiveRecord], dict[str, str]]:
    grouped: dict[str, list[RunRecord]] = defaultdict(list)
    for record in raw_records:
        grouped[record.condition_id].append(record)

    effective_records: list[EffectiveRecord] = []
    selected_raw_to_effective: dict[str, str] = {}

    for condition_id, records in grouped.items():
        selected = max(records, key=record_selection_key)
        source_history = [
            source_history_entry(record, selected.raw_id)
            for record in sorted(records, key=record_selection_key, reverse=True)
        ]
        payload = selected.to_payload_dict()
        payload["is_selected_effective"] = True
        effective = EffectiveRecord(
            effective_id=condition_id,
            condition_id=condition_id,
            selected_raw_id=selected.raw_id,
            selected_suite_name=selected.suite_name,
            status_bucket=selected.status_bucket,
            is_rankable=selected.is_rankable,
            primary_metric_name=selected.primary_metric_name,
            primary_metric_value=selected.primary_metric_value,
            source_history=source_history,
            source_count=len(source_history),
            curve_series=load_curve_series(selected),
            selected_record=payload,
        )
        effective_records.append(effective)
        selected_raw_to_effective[selected.raw_id] = condition_id

    effective_records.sort(
        key=lambda record: (
            record.selected_record.get("dataset_key", ""),
            record.selected_record.get("model", ""),
            record.selected_record.get("split_mode", ""),
            record.selected_record.get("seed", ""),
            record.selected_record.get("condition_id", ""),
        )
    )
    return effective_records, selected_raw_to_effective


def overall_ranking(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_by_dataset: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if not row.get("is_rankable"):
            continue
        model = canonical_text(row.get("model"))
        dataset_key = canonical_text(row.get("dataset_key"))
        current = rows_by_dataset[dataset_key].get(model)
        if current is None or compare_row_strength(row) > compare_row_strength(current):
            rows_by_dataset[dataset_key][model] = row

    contributions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for dataset_key, model_rows in rows_by_dataset.items():
        ranked = sorted(model_rows.values(), key=compare_row_strength, reverse=True)
        if not ranked:
            continue
        participant_count = len(ranked)
        for rank, row in enumerate(ranked, start=1):
            metric = parse_float(row.get("primary_metric_value"))
            if metric is None:
                continue
            contributions[canonical_text(row.get("model"))].append(
                {
                    "dataset_key": dataset_key,
                    "rank": rank,
                    "standing_points": participant_count - rank + 1,
                    "participants": participant_count,
                    "raw_metric": metric,
                }
            )

    ranking_rows: list[dict[str, Any]] = []
    for model, items in contributions.items():
        ranking_rows.append(
            {
                "model": model,
                "datasets_used": len(items),
                "avg_rank": sum(item["rank"] for item in items) / len(items),
                "avg_standing_points": sum(item["standing_points"] for item in items) / len(items),
                "win_count": sum(1 for item in items if item["rank"] == 1),
                "podium_count": sum(1 for item in items if item["rank"] <= min(3, item["participants"])),
            }
        )
    ranking_rows.sort(
        key=lambda row: (
            row["avg_rank"],
            -row["win_count"],
            -row["podium_count"],
            -row["datasets_used"],
            row["model"],
        )
    )
    return ranking_rows


def compare_row_strength(row: dict[str, Any]) -> tuple[float, float, float, int]:
    finished = parse_time(row.get("finished_at"))
    return (
        parse_float(row.get("primary_metric_value")) or float("-inf"),
        parse_float(row.get("selection_metric_value")) or float("-inf"),
        finished.timestamp() if finished else float("-inf"),
        1,
    )


def safe_argument_import_environment() -> None:
    cache_root = Path(tempfile.gettempdir()) / "eeg2_results_import_cache"
    ensure_dir(cache_root)
    ensure_dir(cache_root / "mpl")
    ensure_dir(cache_root / "xdg")
    ensure_dir(cache_root / "mne")
    for key, value in DISCOVERY_ENV_VARS.items():
        os.environ.setdefault(key, value)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "mpl"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    os.environ.setdefault("MNE_HOME", str(cache_root / "mne"))


def action_kind(action: argparse.Action) -> str:
    if isinstance(action, argparse._StoreTrueAction) or isinstance(action, argparse._StoreFalseAction):
        return "bool"
    if action.__class__.__name__ == "BooleanOptionalAction":
        return "bool"
    if getattr(action, "choices", None):
        return "choice"
    if getattr(action, "type", None) in {int, float}:
        return "number"
    return "text"


def action_default_text(action: argparse.Action) -> str:
    if action.default is None or action.default == argparse.SUPPRESS:
        return ""
    return canonical_text(action.default)


def action_choices_text(action: argparse.Action) -> list[str]:
    choices = getattr(action, "choices", None)
    if not choices:
        return []
    return [canonical_text(choice) for choice in choices]


def human_label(dest: str) -> str:
    return dest.replace("_", " ").title()


def fallback_argument_specs() -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    defaults = {
        "model": {},
        "dataset": {},
        "task": {},
        "split_mode": {"choices": ["cs", "ws", "loso"]},
        "fold": {},
        "held_out_subject": {},
        "seed": {},
        "channel_system": {"choices": list(PUBLIC_CHANNEL_SYSTEM_CHOICES)},
        "input_format": {"choices": ["patched", "raw"]},
        "variant": {},
        "epochs": {},
        "batch_size": {},
        "lr": {},
        "encoder_lr_scale": {},
        "classifier_lr_scale": {},
        "epoch_method": {},
        "pipeline": {},
        "patch_size": {},
        "patch_stride": {},
        "desired_time_len": {},
        "model_size": {"choices": ["base", "large"]},
        "depth": {},
    }
    for key, meta in defaults.items():
        specs[key] = {
            "key": key,
            "label": human_label(key),
            "group": "Common",
            "kind": "choice" if meta.get("choices") else "text",
            "help_text": "",
            "default_value": "",
            "choice_values": list(meta.get("choices", [])),
            "models": [],
        }
    return specs


def discover_argument_specs() -> dict[str, dict[str, Any]]:
    safe_argument_import_environment()
    buffer = StringIO()
    try:
        with redirect_stdout(buffer), redirect_stderr(buffer):
            registry_module = importlib.import_module("models.registry")
            run_single_module = importlib.import_module("run_single_experiment")
            list_models = getattr(registry_module, "list_models")
            get_model_config = getattr(registry_module, "get_model_config")
            add_common_args = getattr(run_single_module, "add_common_args")

        specs: dict[str, dict[str, Any]] = {}

        common_parser = argparse.ArgumentParser(add_help=False)
        add_common_args(common_parser)
        common_dests: set[str] = set()
        for action in common_parser._actions:
            if action.dest == "help" or not action.option_strings:
                continue
            common_dests.add(action.dest)
            specs[action.dest] = {
                "key": action.dest,
                "label": human_label(action.dest),
                "group": "Common",
                "kind": action_kind(action),
                "help_text": action.help or "",
                "default_value": action_default_text(action),
                "choice_values": action_choices_text(action),
                "models": [],
            }

        for model in sorted(list_models()):
            parser = argparse.ArgumentParser(add_help=False, conflict_handler="resolve")
            add_common_args(parser)
            config = get_model_config(model)
            config.add_args(parser, mode="finetune")
            for action in parser._actions:
                if action.dest == "help" or not action.option_strings:
                    continue
                if action.dest in common_dests:
                    continue
                group = getattr(getattr(action, "container", None), "title", None) or "Model Specific"
                spec = specs.setdefault(
                    action.dest,
                    {
                        "key": action.dest,
                        "label": human_label(action.dest),
                        "group": group,
                        "kind": action_kind(action),
                        "help_text": action.help or "",
                        "default_value": action_default_text(action),
                        "choice_values": action_choices_text(action),
                        "models": [],
                    },
                )
                if action.help and not spec["help_text"]:
                    spec["help_text"] = action.help
                if action_default_text(action) and not spec["default_value"]:
                    spec["default_value"] = action_default_text(action)
                if action_choices_text(action):
                    spec["choice_values"] = sorted(
                        set(spec["choice_values"]) | set(action_choices_text(action)),
                        key=lambda item: canonical_sort_key(item),
                    )
                if group and spec["group"] == "Model Specific":
                    spec["group"] = group
                if model not in spec["models"]:
                    spec["models"].append(model)
        if specs:
            return specs
    except Exception:
        pass
    return fallback_argument_specs()


def canonical_sort_key(value: Any) -> tuple[int, float | str]:
    text = canonical_text(value)
    number = parse_float(text)
    if number is not None:
        return (0, number)
    return (1, text)


def sort_filter_values(key: str, values: list[str], suite_names: list[str], model_order: list[str], dataset_order: list[str]) -> list[str]:
    custom_orders = {
        "status_bucket": ["completed", "artifacts_missing", "failed", "running", "pending"],
        "split_mode": ["cs", "ws", "loso"],
        "channel_system": list(PUBLIC_CHANNEL_SYSTEM_CHOICES),
        "input_format": ["raw", "patched"],
        "suite_name": suite_names,
        "model": model_order,
        "dataset_key": dataset_order,
    }
    if key in custom_orders:
        order = {value: index for index, value in enumerate(custom_orders[key])}
        return sorted(values, key=lambda item: (order.get(item, 999), canonical_sort_key(item)))
    return sorted(values, key=canonical_sort_key)


def should_exclude_filter_key(key: str) -> bool:
    excluded_keys = {
        "console_log_path",
        "exp_id",
        "run_id",
    }
    return key.endswith("_dir") or key in excluded_keys


def build_filter_schema(
    discovered_specs: dict[str, dict[str, Any]],
    raw_records: list[RunRecord],
    suites: list[SuiteSource],
) -> list[FilterField]:
    observed_values: dict[str, set[str]] = defaultdict(set)
    key_to_models: dict[str, set[str]] = defaultdict(set)
    suite_names = [suite.name for suite in suites]
    model_order = []
    dataset_order = []

    for record in raw_records:
        if record.model and record.model not in model_order:
            model_order.append(record.model)
        if record.dataset_key and record.dataset_key not in dataset_order:
            dataset_order.append(record.dataset_key)
        synthetic_values = {
            "suite_name": record.suite_name,
            "status_bucket": record.status_bucket,
            "dataset_key": record.dataset_key,
            "dataset": record.dataset,
            "task": record.task,
            "model": record.model,
            "split_mode": record.split_mode,
            "channel_system": record.channel_system,
            "classifier_lr_scale": record.classifier_lr_scale,
            "input_format": record.input_format,
            "variant": record.variant,
            "selection_metric_name": record.selection_metric_name,
        }
        for key, value in synthetic_values.items():
            text = canonical_text(value)
            if text:
                observed_values[key].add(text)
        for key, value in record.params.items():
            text = canonical_text(value)
            if text:
                observed_values[key].add(text)
                if record.model:
                    key_to_models[key].add(record.model)
        for key, spec in discovered_specs.items():
            if spec.get("models"):
                key_to_models[key].update(spec["models"])

    synthetic_specs = {
        "suite_name": {"group": "Core", "kind": "choice", "label": "Suite"},
        "status_bucket": {"group": "Core", "kind": "choice", "label": "Status"},
        "dataset_key": {"group": "Core", "kind": "choice", "label": "Dataset"},
        "dataset": {"group": "Core", "kind": "choice", "label": "Dataset Name"},
        "task": {"group": "Core", "kind": "choice", "label": "Task"},
        "model": {"group": "Core", "kind": "choice", "label": "Model"},
        "split_mode": {"group": "Core", "kind": "choice", "label": "Split Mode"},
        "channel_system": {"group": "Core", "kind": "choice", "label": "Channel System"},
        "classifier_lr_scale": {"group": "Core", "kind": "choice", "label": "Classifier LR Scale"},
        "input_format": {"group": "Core", "kind": "choice", "label": "Input Format"},
        "variant": {"group": "Model Scoped", "kind": "choice", "label": "Variant"},
        "selection_metric_name": {"group": "Core", "kind": "choice", "label": "Selection Metric"},
    }

    all_keys = set(discovered_specs) | set(observed_values) | set(synthetic_specs)
    fields: list[FilterField] = []

    for key in all_keys:
        if should_exclude_filter_key(key):
            continue
        spec = discovered_specs.get(key, {})
        synthetic = synthetic_specs.get(key, {})
        raw_group = synthetic.get("group") or spec.get("group") or "Resolved"
        if key.startswith("pipeline_params."):
            raw_group = FILTER_GROUP_OVERRIDES["pipeline_params"]
        elif key.startswith("epoch_params."):
            raw_group = FILTER_GROUP_OVERRIDES["epoch_params"]
        group = FILTER_GROUP_OVERRIDES.get(key, raw_group)
        choices = list(spec.get("choice_values", []))
        kind = synthetic.get("kind") or spec.get("kind") or ("choice" if choices else "text")
        if kind == "bool" and not choices:
            choices = ["true", "false"]
        observed = sort_filter_values(
            key,
            list(observed_values.get(key, set())),
            suite_names=suite_names,
            model_order=model_order,
            dataset_order=dataset_order,
        )
        render_values = observed or sort_filter_values(
            key,
            list({*choices, *( [spec.get("default_value")] if spec.get("default_value") else [])}),
            suite_names=suite_names,
            model_order=model_order,
            dataset_order=dataset_order,
        )
        label = synthetic.get("label") or spec.get("label") or human_label(key)
        default_value = canonical_text(spec.get("default_value"))
        fields.append(
            FilterField(
                key=key,
                label=label,
                group=group,
                kind=kind,
                help_text=canonical_text(spec.get("help_text")),
                default_value=default_value,
                observed_values=observed,
                choice_values=choices,
                render_values=render_values,
                core=key in CORE_FILTER_KEYS,
                enabled=bool(observed),
                disabled_reason="" if observed else "Current results do not include values for this parameter yet.",
                models=sorted(key_to_models.get(key, set())),
            )
        )
    core_priority = {key: index for index, key in enumerate(CORE_FILTER_PRIORITY)}
    fields.sort(
        key=lambda field: (
            0 if field.core else 1,
            core_priority.get(field.key, 999) if field.core else 999,
            field.group.lower(),
            field.label.lower(),
            field.key,
        )
    )
    return fields


def build_summary(
    site_name: str,
    suites: list[SuiteSource],
    raw_rows: list[dict[str, Any]],
    effective_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    raw_status_counts = Counter(row["status_bucket"] for row in raw_rows)
    effective_status_counts = Counter(row["status_bucket"] for row in effective_rows)
    rankable_effective_rows = [row for row in effective_rows if row.get("is_rankable")]
    ranking_rows = overall_ranking(rankable_effective_rows)
    top_model = ranking_rows[0]["model"] if ranking_rows else ""
    top_avg_rank = ranking_rows[0]["avg_rank"] if ranking_rows else None
    top_standing_points = ranking_rows[0]["avg_standing_points"] if ranking_rows else None

    return {
        "site_name": site_name,
        "suite_count": len(suites),
        "raw_run_count": len(raw_rows),
        "effective_run_count": len(effective_rows),
        "rankable_effective_count": len(rankable_effective_rows),
        "completed_count": raw_status_counts.get("completed", 0),
        "artifacts_missing_count": raw_status_counts.get("artifacts_missing", 0),
        "failed_count": raw_status_counts.get("failed", 0),
        "running_count": raw_status_counts.get("running", 0),
        "pending_count": raw_status_counts.get("pending", 0),
        "effective_completed_count": effective_status_counts.get("completed", 0),
        "model_count": len({row["model"] for row in effective_rows if row.get("model")}),
        "dataset_count": len({row["dataset_key"] for row in effective_rows if row.get("dataset_key")}),
        "split_mode_count": len({row["split_mode"] for row in effective_rows if row.get("split_mode")}),
        "suite_names": [suite.name for suite in suites],
        "top_model": top_model,
        "leaderboard_metric": "avg_rank",
        "top_model_avg_rank": top_avg_rank,
        "top_model_standing_points": top_standing_points,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return
    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())
    ordered_keys = [key for key in BASE_FIELD_ORDER if key in all_keys]
    ordered_keys.extend(sorted(all_keys - set(ordered_keys)))

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered_keys)
        writer.writeheader()
        for row in rows:
            serialized = {}
            for key in ordered_keys:
                value = row.get(key)
                if isinstance(value, (dict, list)):
                    serialized[key] = stable_json_text(value)
                elif isinstance(value, bool):
                    serialized[key] = "true" if value else "false"
                elif value is None:
                    serialized[key] = ""
                else:
                    serialized[key] = value
            writer.writerow(serialized)


def render_dashboard(payload: SitePayload) -> str:
    summary = payload.summary
    suites_html = "".join(
        f'<span class="suite-pill" title="{escape(suite["suite_dir"])}">{escape(suite["suite_name"])}</span>'
        for suite in payload.suites
    )
    cards = [
        ("Suites", f'{summary["suite_count"]}'),
        ("Raw Runs", f'{summary["raw_run_count"]:,}'),
        ("Effective Conditions", f'{summary["effective_run_count"]:,}'),
        ("Rankable Conditions", f'{summary["rankable_effective_count"]:,}'),
        ("Models", f'{summary["model_count"]:,}'),
        ("Datasets", f'{summary["dataset_count"]:,}'),
        ("Completed", f'{summary["completed_count"]:,}'),
        ("Failed", f'{summary["failed_count"]:,}'),
    ]
    cards_html = "".join(
        f"""
        <article class="summary-card">
          <div class="summary-kicker">{escape(label)}</div>
          <div class="summary-value">{escape(value)}</div>
        </article>
        """
        for label, value in cards
    )
    data_json = json.dumps(asdict(payload), ensure_ascii=False).replace("</", "<\\/")
    title = escape(payload.site_name)
    generated_at = escape(payload.generated_at)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title} Results Dashboard</title>
  <style>
    :root {{
      --paper: #f2ece0;
      --panel: #fffdf9;
      --ink: #172026;
      --muted: #5c666d;
      --line: #d8d7d0;
      --sea: #154f58;
      --foam: #dff0eb;
      --mint: #6c9d8b;
      --gold: #b98539;
      --rose: #b96b61;
      --shadow: 0 18px 40px rgba(21, 79, 88, 0.09);
      --radius: 22px;
    }}
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; min-height: 100%; }}
    body {{
      background:
        radial-gradient(circle at top right, rgba(21, 79, 88, 0.18), transparent 28%),
        linear-gradient(180deg, #f8f5ef 0%, var(--paper) 100%);
      color: var(--ink);
      font-family: "Trebuchet MS", "Segoe UI", sans-serif;
    }}
    h1, h2, h3 {{ margin: 0; font-family: Georgia, "Times New Roman", serif; }}
    p {{ margin: 0; }}
    button, select {{ font: inherit; }}
    .page {{ width: min(1360px, calc(100vw - 28px)); margin: 18px auto 40px; display: grid; gap: 18px; }}
    .panel {{
      background: rgba(255, 253, 249, 0.96);
      border: 1px solid rgba(216, 215, 208, 0.9);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding: 18px;
      backdrop-filter: blur(10px);
    }}
    .hero {{
      padding: 26px;
      color: white;
      background:
        linear-gradient(135deg, rgba(12, 42, 47, 0.96) 0%, rgba(21, 79, 88, 0.94) 54%, rgba(93, 151, 133, 0.88) 100%);
      overflow: hidden;
      position: relative;
    }}
    .hero::after {{
      content: "";
      position: absolute;
      inset: auto -10% -24% auto;
      width: 340px;
      height: 340px;
      border-radius: 999px;
      background: radial-gradient(circle, rgba(255,255,255,0.25) 0%, transparent 70%);
      pointer-events: none;
    }}
    .hero-top {{ display: grid; gap: 12px; }}
    .hero-copy {{ max-width: none; display: grid; gap: 10px; }}
    .hero h1 {{ font-size: clamp(32px, 4.8vw, 46px); letter-spacing: -0.03em; }}
    .hero-meta {{ display: flex; gap: 10px; flex-wrap: wrap; color: rgba(255,255,255,0.84); font-size: 14px; }}
    .suite-list {{ display: flex; gap: 8px; flex-wrap: wrap; margin-top: 6px; }}
    .suite-pill {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 8px 12px;
      border: 1px solid rgba(255,255,255,0.2);
      border-radius: 999px;
      background: rgba(255,255,255,0.1);
      font-size: 13px;
    }}
    .summary-grid {{ display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); margin-top: 16px; }}
    .summary-card {{
      border-radius: 18px;
      padding: 13px 15px;
      background: rgba(255,255,255,0.1);
      border: 1px solid rgba(255,255,255,0.14);
      min-height: 82px;
    }}
    .summary-kicker {{ font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase; color: rgba(255,255,255,0.76); }}
    .summary-value {{ font-size: 28px; font-weight: 700; margin-top: 8px; }}
    .overview-grid {{
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      margin-top: 16px;
    }}
    .overview-card {{
      border-radius: 20px;
      padding: 14px;
      background: rgba(255,255,255,0.1);
      border: 1px solid rgba(255,255,255,0.14);
      display: grid;
      gap: 10px;
      min-width: 0;
    }}
    .overview-card h3 {{ font-size: 18px; color: white; }}
    .overview-card p {{ line-height: 1.45; color: rgba(255,255,255,0.82); font-size: 12px; }}
    .overview-chart-shell {{
      min-height: 236px;
      border-radius: 16px;
      border: 1px solid rgba(255,255,255,0.12);
      background: rgba(9, 31, 35, 0.12);
      overflow: hidden;
    }}
    .overview-chart-shell svg {{ display: block; width: 100%; height: auto; }}
    .overview-empty {{
      min-height: 236px;
      display: grid;
      place-items: center;
      color: rgba(255,255,255,0.78);
      padding: 22px;
      text-align: center;
    }}
    .overview-grid .legend-row {{ margin-top: 0; }}
    .overview-grid .legend-pill {{
      background: rgba(255,255,255,0.12);
      border-color: rgba(255,255,255,0.18);
      color: rgba(255,255,255,0.9);
    }}
    .overview-grid .legend-swatch {{ box-shadow: inset 0 0 0 1px rgba(255,255,255,0.24); }}
    .overview-footnote {{
      color: rgba(255,255,255,0.78);
      font-size: 12px;
      line-height: 1.4;
      min-height: 17px;
    }}
    .section-head {{ display: flex; gap: 12px; align-items: flex-start; justify-content: space-between; margin-bottom: 14px; flex-wrap: wrap; }}
    .section-head p {{ color: var(--muted); line-height: 1.5; max-width: 860px; }}
    .subtle {{ color: var(--muted); font-size: 14px; }}
    .button-row {{ display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }}
    .ghost-button, .toggle-button, .mini-select {{
      border: 1px solid var(--line);
      background: white;
      border-radius: 999px;
      min-height: 40px;
      padding: 0 14px;
      cursor: pointer;
      color: var(--ink);
    }}
    .ghost-button:hover, .toggle-button:hover, .mini-select:hover {{ border-color: var(--sea); }}
    .toggle-button.active {{
      background: var(--foam);
      color: var(--sea);
      border-color: rgba(21, 79, 88, 0.34);
      font-weight: 600;
    }}
    .page-button.active {{
      background: var(--sea);
      border-color: var(--sea);
      color: white;
      font-weight: 700;
    }}
    .mini-select {{ border-radius: 14px; padding: 8px 12px; min-width: 160px; }}
    .heatmap-toolbar {{
      justify-content: flex-end;
      width: 100%;
    }}
    .heatmap-toolbar .subtle {{
      align-self: center;
      white-space: nowrap;
    }}
    .filter-summary {{ display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 14px; }}
    .filter-chip {{
      padding: 10px 12px;
      border-radius: 14px;
      background: #f6f3ec;
      border: 1px solid var(--line);
      min-width: 140px;
    }}
    .filter-chip .small {{ display: block; color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; }}
    .filter-chip strong {{ display: block; margin-top: 8px; font-size: 24px; }}
    .filter-columns {{ display: grid; gap: 16px; }}
    .filter-group-block {{ display: grid; gap: 14px; }}
    .filter-group-block h3 {{ font-size: 18px; }}
    .filter-toolbar {{ display: grid; gap: 12px; grid-template-columns: repeat(6, minmax(0, 1fr)); }}
    .filter-toolbar > [data-core-filter="dataset_key"] {{
      grid-column: span 2;
    }}
    .compact-filter {{
      border-radius: 16px;
      border: 1px solid var(--line);
      background: #faf7f0;
      padding: 12px;
      display: grid;
      gap: 10px;
      align-content: start;
      min-height: 88px;
    }}
    .compact-filter.priority {{
      background: #fffefb;
      border-color: rgba(21, 79, 88, 0.18);
    }}
    .compact-filter.is-disabled {{
      background: #f7f4ef;
      color: var(--muted);
    }}
    .compact-filter-head {{
      display: flex;
      gap: 10px;
      align-items: baseline;
      justify-content: space-between;
      min-height: 18px;
    }}
    .compact-filter label {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      white-space: nowrap;
      margin: 0;
    }}
    .compact-select {{
      width: 100%;
      border-radius: 12px;
      border: 1px solid var(--line);
      min-height: 40px;
      padding: 8px 10px;
      background: white;
      color: var(--ink);
    }}
    .compact-help {{
      color: var(--muted);
      font-size: 12px;
      line-height: 1.2;
      white-space: nowrap;
      flex: 0 0 auto;
      text-align: right;
    }}
    .multi-select-dropdown {{ position: relative; }}
    .multi-select-dropdown summary {{
      list-style: none;
      cursor: pointer;
    }}
    .multi-select-dropdown summary::-webkit-details-marker {{ display: none; }}
    .multi-select-summary {{
      width: 100%;
      border-radius: 12px;
      border: 1px solid var(--line);
      min-height: 40px;
      padding: 8px 10px;
      background: white;
      color: var(--ink);
      display: flex;
      gap: 10px;
      align-items: center;
      justify-content: space-between;
      user-select: none;
    }}
    .multi-select-summary:hover,
    .multi-select-dropdown[open] .multi-select-summary {{
      border-color: var(--sea);
    }}
    .multi-select-summary-text {{
      min-width: 0;
      display: grid;
      gap: 2px;
    }}
    .multi-select-summary-value {{
      font-size: 14px;
      font-weight: 600;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .multi-select-summary-meta {{
      font-size: 11px;
      color: var(--muted);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .multi-select-caret {{
      flex: 0 0 auto;
      color: var(--muted);
      transition: transform 120ms ease;
    }}
    .multi-select-dropdown[open] .multi-select-caret {{
      transform: rotate(180deg);
    }}
    .multi-select-panel {{
      position: absolute;
      top: calc(100% + 8px);
      left: 0;
      right: 0;
      z-index: 40;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: white;
      box-shadow: 0 16px 34px rgba(19, 36, 40, 0.16);
      padding: 10px;
    }}
    .multi-select-scroll {{
      max-height: 264px;
      overflow: auto;
      display: grid;
      gap: 6px;
    }}
    .filter-inline-options {{ display: flex; gap: 6px; flex-wrap: wrap; }}
    .filter-inline-options.compact-row {{
      flex-wrap: nowrap;
      overflow-x: auto;
      padding-bottom: 2px;
      scrollbar-width: thin;
    }}
    .filter-grid {{ display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); }}
    .filter-grid > [data-filter-card="pipeline"] {{
      grid-column: span 2;
    }}
    .filter-card {{
      border-radius: 18px;
      border: 1px solid var(--line);
      background: #fffefb;
      padding: 14px;
      display: grid;
      gap: 10px;
      align-content: start;
    }}
    .filter-card.is-disabled {{
      background: #f7f4ef;
      color: var(--muted);
    }}
    .filter-title-row {{ display: flex; gap: 8px; align-items: center; justify-content: space-between; }}
    .filter-title-row h4 {{ font-size: 16px; margin: 0; }}
    .group-tag {{
      font-size: 11px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
      background: #f1eee7;
      padding: 5px 8px;
      border-radius: 999px;
      border: 1px solid var(--line);
    }}
    .filter-options {{ display: flex; gap: 8px; flex-wrap: wrap; }}
    .filter-option {{
      border-radius: 999px;
      border: 1px solid var(--line);
      background: white;
      color: var(--ink);
      padding: 8px 12px;
      cursor: pointer;
      transition: border-color 120ms ease, background 120ms ease, color 120ms ease;
      max-width: 100%;
      overflow-wrap: anywhere;
    }}
    .filter-option.compact {{
      white-space: nowrap;
      flex: 0 0 auto;
      padding-inline: 11px;
    }}
    .filter-option.dropdown-option {{
      width: 100%;
      border-radius: 12px;
      display: flex;
      justify-content: flex-start;
      text-align: left;
      white-space: normal;
      line-height: 1.35;
      padding: 10px 12px;
    }}
    .filter-option.active {{
      background: var(--sea);
      border-color: var(--sea);
      color: white;
    }}
    .filter-option.any.active {{
      background: var(--foam);
      border-color: rgba(21, 79, 88, 0.34);
      color: var(--sea);
    }}
    .filter-option:disabled {{
      cursor: not-allowed;
      color: #999;
      background: #f3f0ea;
    }}
    .compact-filter.is-disabled .multi-select-summary {{
      cursor: not-allowed;
      background: #f3f0ea;
      color: #999;
    }}
    .compact-filter.is-disabled .multi-select-dropdown summary {{
      pointer-events: none;
    }}
    .filter-meta {{ color: var(--muted); font-size: 13px; line-height: 1.45; }}
    .details-toggle {{
      border: 1px solid var(--line);
      border-radius: 18px;
      background: #faf7f1;
      padding: 12px 14px;
    }}
    .details-toggle summary {{ cursor: pointer; font-weight: 600; }}
    .details-toggle[open] summary {{ margin-bottom: 14px; }}
    .chart-shell {{
      border-radius: 18px;
      border: 1px solid var(--line);
      background: linear-gradient(180deg, #fffefb 0%, #f6f4ee 100%);
      min-height: 320px;
      overflow: auto;
      display: grid;
      align-items: stretch;
    }}
    .chart-shell svg {{ display: block; height: auto; max-width: none; }}
    .chart-empty {{
      min-height: 320px;
      display: grid;
      place-items: center;
      color: var(--muted);
      padding: 28px;
      text-align: center;
    }}
    .legend-row {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 12px;
    }}
    .legend-pill {{
      display: inline-flex;
      align-items: center;
      gap: 7px;
      border-radius: 999px;
      border: 1px solid var(--line);
      padding: 6px 10px;
      background: #fff;
      color: var(--muted);
      font-size: 12px;
    }}
    .legend-swatch {{ width: 10px; height: 10px; border-radius: 999px; display: inline-block; }}
    .compare-grid {{ display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }}
    .subsection-head {{ display: flex; gap: 12px; align-items: flex-start; justify-content: space-between; margin: 18px 0 12px; flex-wrap: wrap; }}
    .subsection-head p {{ color: var(--muted); line-height: 1.5; max-width: 820px; }}
    .leaderboard-grid {{
      display: grid;
      gap: 16px;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      align-items: start;
    }}
    .leaderboard-column {{
      border-radius: 22px;
      border: 1px solid var(--line);
      background: linear-gradient(180deg, #fffdf8 0%, #f6f2e8 100%);
      padding: 12px;
      display: grid;
      gap: 10px;
      min-height: 280px;
      min-width: 0;
    }}
    .leaderboard-column-head {{
      display: flex;
      gap: 10px;
      align-items: flex-end;
      justify-content: space-between;
      padding: 4px 2px 0;
    }}
    .leaderboard-column-head h3 {{
      margin: 0;
      font-size: 24px;
      letter-spacing: 0.08em;
    }}
    .leaderboard-column-head p {{
      margin: 0;
      color: var(--muted);
      font-size: 13px;
    }}
    .leaderboard-stack {{ display: grid; gap: 6px; }}
    .leaderboard-empty {{
      min-height: 240px;
      border-radius: 18px;
      border: 1px dashed #d9d4ca;
      display: grid;
      place-items: center;
      color: var(--muted);
      background: rgba(255,255,255,0.58);
      text-align: center;
      padding: 20px;
    }}
    .leaderboard-bar {{
      position: relative;
      display: block;
      width: 100%;
      min-height: 32px;
      padding: 5px 12px 5px 38px;
      border-radius: 14px;
      border: 1px solid #d8d2c7;
      background: white;
      overflow: hidden;
      text-align: left;
      cursor: pointer;
      transition: transform 120ms ease, box-shadow 120ms ease, border-color 120ms ease;
    }}
    .leaderboard-bar:hover {{
      transform: translateY(-1px);
      box-shadow: 0 12px 24px rgba(25, 33, 41, 0.08);
    }}
    .leaderboard-bar::before {{
      content: "";
      position: absolute;
      inset: 0;
      background: linear-gradient(
        90deg,
        var(--bar-soft) 0%,
        var(--bar-soft) var(--bar-fill),
        rgba(255,255,255,0.92) var(--bar-fill),
        rgba(255,255,255,0.98) 100%
      );
      z-index: 0;
    }}
    .leaderboard-bar::after {{
      content: "";
      position: absolute;
      inset: 0 auto 0 0;
      width: 5px;
      background: var(--bar-color);
      z-index: 1;
    }}
    .leaderboard-bar.is-selected {{
      border-color: var(--bar-color);
      box-shadow: 0 0 0 2px var(--bar-color), 0 16px 32px rgba(25, 33, 41, 0.12);
    }}
    .leaderboard-bar.is-linked {{
      border-color: var(--bar-color);
      box-shadow: inset 0 0 0 1px var(--bar-color);
    }}
    .leaderboard-bar.is-muted {{
      opacity: 0.4;
    }}
    .leaderboard-bar.is-hovered {{
      border-color: var(--bar-color);
      box-shadow: 0 0 0 2px var(--bar-color), 0 14px 28px rgba(25, 33, 41, 0.12);
    }}
    .leaderboard-rank-badge {{
      position: absolute;
      left: 10px;
      top: 50%;
      transform: translateY(-50%);
      z-index: 2;
      width: 18px;
      height: 18px;
      border-radius: 999px;
      display: grid;
      place-items: center;
      background: rgba(255,255,255,0.88);
      border: 1px solid rgba(0,0,0,0.08);
      color: var(--bar-deep);
      font-size: 9px;
      font-weight: 700;
    }}
    .leaderboard-content {{
      position: relative;
      z-index: 2;
      display: grid;
      gap: 0;
      min-width: 0;
    }}
    .leaderboard-topline {{
      display: flex;
      gap: 10px;
      align-items: center;
      justify-content: space-between;
      flex-wrap: nowrap;
      min-width: 0;
    }}
    .leaderboard-model {{
      flex: 0 1 auto;
      min-width: 0;
      max-width: 34%;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-size: 14px;
      font-weight: 700;
      color: var(--ink);
      letter-spacing: -0.01em;
      line-height: 1.1;
    }}
    .leaderboard-stats {{
      flex: 1 1 auto;
      min-width: 0;
      display: flex;
      gap: 9px;
      justify-content: flex-end;
      align-items: center;
      flex-wrap: nowrap;
      overflow: hidden;
      color: #24323a;
      font-size: 11px;
      line-height: 1.2;
    }}
    .leaderboard-stat {{
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .leaderboard-stat.metric {{ color: #455861; font-weight: 600; }}
    .leaderboard-column-curve {{ display: grid; gap: 8px; min-width: 0; }}
    .split-curve-shell {{ min-height: 276px; overflow: hidden; }}
    .split-curve-shell .chart-empty {{ min-height: 276px; }}
    .split-curve-series [data-curve-main],
    .split-curve-series [data-curve-underlay],
    .split-curve-series [data-curve-point] {{
      transition: opacity 120ms ease, stroke-width 120ms ease, r 120ms ease;
    }}
    .split-curve-series.is-muted [data-curve-main] {{ opacity: 0.1 !important; }}
    .split-curve-series.is-muted [data-curve-point] {{ opacity: 0.08 !important; }}
    .split-curve-series.is-muted [data-curve-underlay] {{ opacity: 0 !important; }}
    .split-curve-series.is-linked [data-curve-underlay] {{ opacity: 0.3 !important; }}
    .split-curve-series.is-linked [data-curve-main] {{ stroke-width: 3.2 !important; opacity: 0.96 !important; }}
    .split-curve-series.is-linked [data-curve-point] {{ opacity: 0.92 !important; }}
    .split-curve-series.is-selected [data-curve-underlay] {{ opacity: 0.94 !important; }}
    .split-curve-series.is-selected [data-curve-main] {{ stroke-width: 4.3 !important; opacity: 1 !important; }}
    .split-curve-series.is-selected [data-curve-point] {{ opacity: 1 !important; }}
    .split-curve-series.is-hovered [data-curve-underlay] {{ opacity: 0.96 !important; }}
    .split-curve-series.is-hovered [data-curve-main] {{ stroke-width: 4.6 !important; opacity: 1 !important; }}
    .split-curve-series.is-hovered [data-curve-point] {{ opacity: 1 !important; }}
    .drilldown-layout {{ display: grid; gap: 18px; min-width: 0; }}
    .drilldown-main, .drilldown-aside {{
      display: grid;
      gap: 14px;
      align-content: start;
      min-width: 0;
    }}
    .drilldown-aside {{ grid-template-columns: minmax(240px, 320px) minmax(0, 1fr); }}
    .drilldown-toolbar {{ display: grid; gap: 12px; grid-template-columns: minmax(0, 1.25fr) minmax(180px, 0.55fr) minmax(260px, 1fr); min-width: 0; }}
    .control-grid {{ display: grid; gap: 12px; }}
    .control-card {{
      border-radius: 18px;
      border: 1px solid var(--line);
      background: #faf7f0;
      padding: 14px;
      display: grid;
      gap: 10px;
      min-width: 0;
    }}
    .control-card label {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    .control-card select {{
      width: 100%;
      border-radius: 14px;
      border: 1px solid var(--line);
      min-height: 42px;
      padding: 8px 12px;
      background: white;
    }}
    .selection-card {{ background: linear-gradient(180deg, #fbf8f1 0%, #f4efe4 100%); }}
    .selection-summary {{ display: grid; gap: 8px; }}
    .selection-summary .small {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .selection-summary strong {{
      font-size: 28px;
      line-height: 1.05;
      letter-spacing: -0.02em;
    }}
    .selection-pills {{ display: flex; gap: 8px; flex-wrap: wrap; }}
    .selection-pill {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.86);
      padding: 6px 10px;
      color: #45535c;
      font-size: 12px;
    }}
    .explorer-kpis {{ display: grid; gap: 10px; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); }}
    .explorer-kpi {{
      border-radius: 16px;
      border: 1px solid var(--line);
      padding: 12px;
      background: #fff;
    }}
    .explorer-kpi .small {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; }}
    .explorer-kpi strong {{ display: block; margin-top: 8px; font-size: 24px; }}
    .detail-grid {{ display: grid; gap: 14px; grid-template-columns: 1fr 1fr; }}
    .detail-list {{
      border-radius: 16px;
      border: 1px solid var(--line);
      background: #faf7f0;
      padding: 14px;
      display: grid;
      gap: 10px;
      min-width: 0;
    }}
    .detail-list h3 {{ margin: 0; }}
    .dataset-card-grid {{ display: grid; gap: 10px; }}
    .dataset-card {{
      width: 100%;
      border-radius: 16px;
      border: 1px solid var(--line);
      background: white;
      padding: 12px 14px 12px 16px;
      text-align: left;
      position: relative;
      overflow: hidden;
      cursor: pointer;
      transition: border-color 120ms ease, transform 120ms ease, box-shadow 120ms ease;
    }}
    .dataset-card::before {{
      content: "";
      position: absolute;
      inset: 0 auto 0 0;
      width: 5px;
      background: var(--dataset-color, #6f8aa1);
    }}
    .dataset-card:hover {{
      transform: translateY(-1px);
      box-shadow: 0 10px 22px rgba(25, 33, 41, 0.08);
    }}
    .dataset-card.selected {{
      border-color: var(--dataset-color, #6f8aa1);
      box-shadow: 0 0 0 2px var(--dataset-color, #6f8aa1);
      background: linear-gradient(180deg, #fffdfa 0%, #f7f3eb 100%);
    }}
    .dataset-card-head {{
      display: flex;
      gap: 10px;
      justify-content: space-between;
      align-items: baseline;
      flex-wrap: wrap;
    }}
    .dataset-card-head strong {{ font-size: 15px; }}
    .dataset-card-metric {{
      color: #24323a;
      font-size: 13px;
      font-weight: 600;
    }}
    .dataset-card-meta {{
      margin-top: 7px;
      color: var(--muted);
      font-size: 12.5px;
      line-height: 1.45;
    }}
    .history-list {{
      display: grid;
      gap: 8px;
      max-height: 420px;
      overflow: auto;
      padding-right: 2px;
    }}
    .history-item {{
      border-radius: 14px;
      padding: 10px 12px;
      background: white;
      border: 1px solid var(--line);
    }}
    .history-item.selected {{
      border-color: rgba(21, 79, 88, 0.36);
      background: var(--foam);
    }}
    .table-wrap {{
      border-radius: 18px;
      border: 1px solid var(--line);
      overflow: auto;
      background: #fff;
      min-width: 0;
    }}
    table {{ width: 100%; border-collapse: collapse; min-width: 620px; }}
    th, td {{ padding: 11px 12px; border-bottom: 1px solid #ece8de; text-align: left; vertical-align: top; }}
    th {{
      background: #f6f3ec;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    td {{ font-size: 14px; }}
    .mono {{ font-family: Consolas, Monaco, monospace; font-size: 13px; }}
    .footer-note {{ color: var(--muted); font-size: 13px; line-height: 1.45; }}
    @media (max-width: 1120px) {{
      .overview-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .filter-toolbar {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }}
      .leaderboard-grid, .drilldown-aside, .drilldown-toolbar, .detail-grid {{ grid-template-columns: 1fr; }}
    }}
    @media (max-width: 900px) {{
      .filter-grid > [data-filter-card="pipeline"] {{ grid-column: auto; }}
    }}
    @media (max-width: 760px) {{
      .page {{ width: min(100vw - 18px, 1360px); margin: 10px auto 24px; }}
      .panel, .hero {{ padding: 16px; }}
      .summary-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .overview-grid {{ grid-template-columns: 1fr; }}
      .filter-toolbar {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .filter-toolbar > [data-core-filter="dataset_key"] {{ grid-column: 1 / -1; }}
      .leaderboard-bar {{ min-height: 40px; padding: 6px 10px 6px 38px; }}
      .leaderboard-topline {{ gap: 6px; flex-wrap: wrap; align-items: flex-start; }}
      .leaderboard-model {{ max-width: none; }}
      .leaderboard-stats {{ justify-content: flex-start; flex-wrap: wrap; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="panel hero">
      <div class="hero-top">
        <div class="hero-copy">
          <div class="hero-meta">
            <span>Generated at {generated_at}</span>
            <span>{escape(str(summary["suite_count"]))} suite(s)</span>
            <span>{escape(str(summary["raw_run_count"]))} raw runs</span>
            <span>{escape(str(summary["effective_run_count"]))} effective conditions</span>
          </div>
          <h1>{title}</h1>
          <div class="suite-list">{suites_html}</div>
        </div>
      </div>
      <div class="summary-grid">{cards_html}</div>
      <div class="overview-grid">
        <article class="overview-card">
          <div>
            <h3>Run Status</h3>
            <p>Global raw-run status mix for the current dashboard build.</p>
          </div>
          <div id="overview-status" class="overview-chart-shell"></div>
          <div id="overview-status-legend" class="legend-row"></div>
        </article>
        <article class="overview-card">
          <div>
            <h3>Runs by Split Mode</h3>
            <p>Raw-run coverage by split mode, keeping empty names grouped as Unknown.</p>
          </div>
          <div id="overview-split" class="overview-chart-shell"></div>
          <div id="overview-split-note" class="overview-footnote"></div>
        </article>
        <article class="overview-card">
          <div>
            <h3>Runs by Suite</h3>
            <p>Status-stacked suite totals so the current batch health is visible at a glance.</p>
          </div>
          <div id="overview-suite" class="overview-chart-shell"></div>
          <div id="overview-suite-legend" class="legend-row"></div>
        </article>
      </div>
    </section>

    <section class="panel">
      <div class="section-head">
        <div>
          <h2>Parameter Filters</h2>
          <p>All discovered parameters appear below. Active values filter the leaderboard, heatmap, and dataset drilldown together. Parameters with no observed values are still shown, but disabled.</p>
        </div>
        <div class="button-row">
          <button id="reset-filters" class="ghost-button" type="button">Reset Filters</button>
        </div>
      </div>
      <div class="filter-summary">
        <div class="filter-chip"><span class="small">Matching Raw Runs</span><strong id="matching-raw">0</strong></div>
        <div class="filter-chip"><span class="small">Matching Effective</span><strong id="matching-effective">0</strong></div>
        <div class="filter-chip"><span class="small">Rankable Effective</span><strong id="matching-rankable">0</strong></div>
      </div>
      <div class="filter-columns">
        <div class="filter-group-block">
          <h3>Core Filters</h3>
          <div id="core-filters" class="filter-toolbar"></div>
        </div>
        <details class="details-toggle">
          <summary>Model Parameters</summary>
          <div id="model-parameter-filters" class="filter-grid"></div>
        </details>
        <details class="details-toggle">
          <summary>Advanced Parameters</summary>
          <div id="advanced-filters" class="filter-grid"></div>
        </details>
      </div>
    </section>

    <section class="panel">
      <div class="section-head">
        <div>
          <h2>Overall Leaderboard</h2>
          <p>This is the primary summary view. Each split keeps its own leaderboard and its own curve chart. Filters decide which results stay visible, while hover highlights the matching model between bars and curves.</p>
        </div>
        <div class="button-row">
          <label class="subtle" for="split-curve-mode">Curve Mode</label>
          <select id="split-curve-mode" class="mini-select">
            <option value="val_primary">Validation Primary</option>
            <option value="test_primary">Test Primary</option>
            <option value="train_loss">Train Loss</option>
            <option value="lr">Learning Rate</option>
          </select>
        </div>
      </div>
      <div id="overall-ranking" class="leaderboard-grid"></div>
      <p id="overall-ranking-note" class="footer-note"></p>
    </section>

    <section class="panel">
      <div class="section-head">
        <div>
          <h2>Model × Dataset Heatmap</h2>
          <p>Cells show the best filtered effective run for each model and dataset. Models are paged in fixed-size slices so larger suites stay readable without stretching the page.</p>
        </div>
        <div id="heatmap-pagination" class="button-row heatmap-toolbar"></div>
      </div>
      <div id="heatmap" class="chart-shell"></div>
    </section>
  </main>

  <script id="app-data" type="application/json">{data_json}</script>
  <script>
    const APP = JSON.parse(document.getElementById("app-data").textContent);
    const STATUS_LABELS = {json.dumps(STATUS_LABELS, ensure_ascii=False)};
    const STATUS_ORDER = ["completed", "artifacts_missing", "failed", "running", "pending"];
    const HEATMAP_MODELS_PER_PAGE = 6;
    const STATUS_COLORS = {{
      completed: "#2b8a78",
      artifacts_missing: "#c38a32",
      failed: "#bf5a52",
      running: "#4a77a8",
      pending: "#8f939a",
      unknown: "#70757c",
    }};
    const FILTER_FIELDS = APP.filters;
    const EFFECTIVE_ROWS = APP.effective_runs;
    const RAW_ROWS = APP.raw_runs;
    const MODEL_PALETTE = ["#005f73", "#bb3e03", "#0a9396", "#7b2cbf", "#ca6702", "#386641", "#1d4ed8", "#c1121f", "#2a9d8f", "#6a4c93", "#d97706", "#2b6cb0"];
    const DATASET_PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#8C564B", "#7F3C8D", "#C44E52", "#2CA02C"];
    const SPLIT_PALETTE = {{
      ws: "#2f5d8b",
      cs: "#b85b3f",
      loso: "#5f7d54",
    }};

    const STATE = {{
      filters: {{}},
      heatmapPage: 0,
      leaderboardSplitMode: "",
      leaderboardModel: "",
      explorerCurveMode: "val_primary",
      hoverSplitMode: "",
      hoverModel: "",
    }};

    function esc(value) {{
      return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
    }}

    function num(value) {{
      const parsed = Number(value);
      return Number.isFinite(parsed) ? parsed : null;
    }}

    function fmt(value, digits = 3) {{
      const parsed = num(value);
      return parsed === null ? "-" : parsed.toLocaleString("en-US", {{ minimumFractionDigits: digits, maximumFractionDigits: digits }});
    }}

    function fmtDate(value) {{
      if (!value) return "-";
      return value.replace("T", " ").replace("+08:00", "");
    }}

    function sortByOrder(values, preferred) {{
      const order = new Map(preferred.map((value, index) => [value, index]));
      return [...values].sort((a, b) => {{
        const left = order.has(a) ? order.get(a) : Number.MAX_SAFE_INTEGER;
        const right = order.has(b) ? order.get(b) : Number.MAX_SAFE_INTEGER;
        if (left !== right) return left - right;
        const leftNum = num(a);
        const rightNum = num(b);
        if (leftNum !== null && rightNum !== null) return leftNum - rightNum;
        return String(a).localeCompare(String(b), undefined, {{ numeric: true, sensitivity: "base" }});
      }});
    }}

    function hashValue(seed) {{
      let total = 0;
      for (let index = 0; index < seed.length; index += 1) {{
        total += seed.charCodeAt(index) * (index + 17);
      }}
      return total;
    }}

    function paletteColor(seed, palette) {{
      return palette[hashValue(seed) % palette.length];
    }}

    function modelColor(model) {{
      return paletteColor(`model:${{model}}`, MODEL_PALETTE);
    }}

    function datasetColor(datasetKey) {{
      return paletteColor(`dataset:${{datasetKey}}`, DATASET_PALETTE);
    }}

    function splitColor(splitMode) {{
      const key = String(splitMode || "").toLowerCase();
      return SPLIT_PALETTE[key] || paletteColor(`split:${{key}}`, MODEL_PALETTE);
    }}

    function clamp(value, minimum, maximum) {{
      return Math.min(maximum, Math.max(minimum, value));
    }}

    function shortLabel(value, limit = 32) {{
      const text = String(value ?? "");
      return text.length <= limit ? text : `${{text.slice(0, Math.max(0, limit - 1))}}…`;
    }}

    function splitLabel(splitMode) {{
      return String(splitMode || "").toUpperCase() || "-";
    }}

    function optionCountLabel(count) {{
      return `${{count}} option${{count === 1 ? "" : "s"}}`;
    }}

    function datasetFilterLabel(value) {{
      return String(value || "").replace(":", " / ");
    }}

    function datasetFilterSummary(field) {{
      const values = field.render_values || [];
      const selectedSet = STATE.filters[field.key] || new Set();
      const selectedLabels = values
        .filter((value) => selectedSet.has(value))
        .map((value) => datasetFilterLabel(value));
      if (!selectedLabels.length) {{
        return {{
          value: "Any dataset",
          meta: "All datasets included",
          title: "Any dataset",
        }};
      }}
      if (selectedLabels.length === 1) {{
        return {{
          value: shortLabel(selectedLabels[0], 34),
          meta: "1 dataset selected",
          title: selectedLabels[0],
        }};
      }}
      const preview = selectedLabels.length === 2
        ? `${{shortLabel(selectedLabels[0], 20)}} + 1 more`
        : `${{shortLabel(selectedLabels[0], 20)}} + ${{selectedLabels.length - 1}} more`;
      return {{
        value: `${{selectedLabels.length}} datasets selected`,
        meta: preview,
        title: selectedLabels.join(", "),
      }};
    }}

    function svgFrame(width, height, label) {{
      return `<svg viewBox="0 0 ${{width}} ${{height}}" role="img" aria-label="${{esc(label)}}" style="width:100%;min-width:${{width}}px;height:auto;display:block">`;
    }}

    function responsiveSvgFrame(width, height, label) {{
      return `<svg viewBox="0 0 ${{width}} ${{height}}" role="img" aria-label="${{esc(label)}}" style="width:100%;height:auto;display:block">`;
    }}

    function hexToRgb(hex) {{
      const normalized = String(hex).replace("#", "");
      const text = normalized.length === 3 ? normalized.split("").map((char) => char + char).join("") : normalized;
      return {{
        r: Number.parseInt(text.slice(0, 2), 16),
        g: Number.parseInt(text.slice(2, 4), 16),
        b: Number.parseInt(text.slice(4, 6), 16),
      }};
    }}

    function rgbToHex(color) {{
      const encode = (value) => clamp(Math.round(value), 0, 255).toString(16).padStart(2, "0");
      return `#${{encode(color.r)}}${{encode(color.g)}}${{encode(color.b)}}`;
    }}

    function mixColor(left, right, ratio) {{
      const t = clamp(ratio, 0, 1);
      const a = hexToRgb(left);
      const b = hexToRgb(right);
      return rgbToHex({{
        r: a.r + (b.r - a.r) * t,
        g: a.g + (b.g - a.g) * t,
        b: a.b + (b.b - a.b) * t,
      }});
    }}

    function triBlend(low, middle, high, ratio) {{
      const t = clamp(ratio, 0, 1);
      return t <= 0.5 ? mixColor(low, middle, t * 2) : mixColor(middle, high, (t - 0.5) * 2);
    }}

    function heatTone(intensity) {{
      return triBlend("#fbf6ed", "#d6e5e8", "#315c7c", intensity ?? 0);
    }}

    function confusionTone(intensity) {{
      return triBlend("#fbf6ed", "#dbe8df", "#4b5d8a", intensity ?? 0);
    }}

    function normalizeGroupLabel(value) {{
      const text = String(value ?? "").trim();
      return text ? text : "Unknown";
    }}

    function normalizeStatusBucket(value) {{
      const key = String(value ?? "").trim().toLowerCase();
      return STATUS_ORDER.includes(key) ? key : (key ? key : "unknown");
    }}

    function statusColor(status) {{
      return STATUS_COLORS[status] || STATUS_COLORS.unknown;
    }}

    function statusLabel(status) {{
      return status === "unknown" ? "Unknown" : (STATUS_LABELS[status] || normalizeGroupLabel(status));
    }}

    function chartWidth(targetId, fallback = 560, minimum = 320, maximum = 860) {{
      const target = document.getElementById(targetId);
      const measured = target ? Math.round(target.getBoundingClientRect().width || target.clientWidth || fallback) : fallback;
      return clamp(measured || fallback, minimum, maximum);
    }}

    function metricKey(row) {{
      return row.primary_metric_name || APP.dataset_primary_metrics[row.dataset_key] || "acc";
    }}

    function compareStrength(row) {{
      return {{
        primary: num(row.primary_metric_value) ?? Number.NEGATIVE_INFINITY,
        selection: num(row.selection_metric_value) ?? Number.NEGATIVE_INFINITY,
        finished: row.finished_at ? Date.parse(row.finished_at) : Number.NEGATIVE_INFINITY,
        suiteIndex: num(row.suite_index) ?? 0,
      }};
    }}

    function compareRows(left, right) {{
      const a = compareStrength(left);
      const b = compareStrength(right);
      if (a.primary !== b.primary) return b.primary - a.primary;
      if (a.selection !== b.selection) return b.selection - a.selection;
      if (a.finished !== b.finished) return b.finished - a.finished;
      return b.suiteIndex - a.suiteIndex;
    }}

    function anyActive(key) {{
      const selected = STATE.filters[key];
      return selected && selected.size > 0;
    }}

    function rowMatchesFilters(row) {{
      return FILTER_FIELDS.every((field) => {{
        const selected = STATE.filters[field.key];
        if (!selected || selected.size === 0) return true;
        return selected.has(String(row[field.key] ?? ""));
      }});
    }}

    function filteredEffectiveRows() {{
      return EFFECTIVE_ROWS.filter((row) => rowMatchesFilters(row));
    }}

    function filteredRawRows() {{
      return RAW_ROWS.filter((row) => rowMatchesFilters(row));
    }}

    function filteredRankableRows() {{
      return filteredEffectiveRows().filter((row) => row.is_rankable);
    }}

    function datasetRankIntensityMap(rows) {{
      const grouped = new Map();
      rows.forEach((row) => {{
        if (!row.is_rankable) return;
        if (!grouped.has(row.dataset_key)) grouped.set(row.dataset_key, []);
        grouped.get(row.dataset_key).push(row);
      }});
      const result = new Map();
      grouped.forEach((datasetRows) => {{
        const ranked = bestByModel(datasetRows);
        const total = ranked.length;
        ranked.forEach((row, index) => {{
          const intensity = total <= 1 ? 1 : 1 - index / (total - 1);
          result.set(row.effective_id, intensity);
        }});
      }});
      return result;
    }}

    function bestByModel(rows) {{
      const byModel = new Map();
      rows.forEach((row) => {{
        if (!row.is_rankable) return;
        const key = row.model;
        const current = byModel.get(key);
        if (!current || compareRows(row, current) < 0) {{
          byModel.set(key, row);
        }}
      }});
      return [...byModel.values()].sort(compareRows);
    }}

    function bestByDataset(rows) {{
      const byDataset = new Map();
      rows.forEach((row) => {{
        if (!row.is_rankable) return;
        const key = row.dataset_key;
        const current = byDataset.get(key);
        if (!current || compareRows(row, current) < 0) {{
          byDataset.set(key, row);
        }}
      }});
      const order = new Map(APP.dataset_order.map((datasetKey, index) => [datasetKey, index]));
      return [...byDataset.values()].sort((left, right) => {{
        const leftOrder = order.has(left.dataset_key) ? order.get(left.dataset_key) : Number.MAX_SAFE_INTEGER;
        const rightOrder = order.has(right.dataset_key) ? order.get(right.dataset_key) : Number.MAX_SAFE_INTEGER;
        if (leftOrder !== rightOrder) return leftOrder - rightOrder;
        return left.dataset_key.localeCompare(right.dataset_key);
      }});
    }}

    function datasetRanking(rows, datasetKey) {{
      return bestByModel(rows.filter((row) => row.dataset_key === datasetKey));
    }}

    function overallRankingRows(rows) {{
      const grouped = new Map();
      rows.forEach((row) => {{
        if (!row.is_rankable) return;
        if (!grouped.has(row.dataset_key)) grouped.set(row.dataset_key, []);
        grouped.get(row.dataset_key).push(row);
      }});

      const contributions = new Map();
      grouped.forEach((datasetRows, datasetKey) => {{
        const ranked = bestByModel(datasetRows);
        const participants = ranked.length;
        if (!participants) return;
        ranked.forEach((row, index) => {{
          const value = num(row.primary_metric_value);
          const selectionValue = num(row.selection_metric_value);
          const entry = contributions.get(row.model) || [];
          entry.push({{
            dataset_key: datasetKey,
            rank: index + 1,
            standing_points: participants - index,
            participants,
            raw_metric: value,
            selection_metric_value: selectionValue,
            selection_metric_name: row.selection_metric_name,
            row,
          }});
          contributions.set(row.model, entry);
        }});
      }});

      return [...contributions.entries()]
        .map(([model, items]) => {{
          const avgRank = items.reduce((acc, item) => acc + item.rank, 0) / items.length;
          const avgStandingPoints = items.reduce((acc, item) => acc + item.standing_points, 0) / items.length;
          const winCount = items.filter((item) => item.rank === 1).length;
          const podiumCount = items.filter((item) => item.rank <= Math.min(3, item.participants)).length;
          const rawValues = items.map((item) => item.raw_metric).filter((value) => value !== null);
          const primaryNames = [...new Set(items.map((item) => item.row?.primary_metric_name).filter(Boolean))];
          const selectionValues = items.map((item) => item.selection_metric_value).filter((value) => value !== null);
          const selectionNames = [...new Set(items.map((item) => item.selection_metric_name).filter(Boolean))];
          return {{
            model,
            items,
            datasets_used: items.length,
            avg_rank: avgRank,
            avg_standing_points: avgStandingPoints,
            win_count: winCount,
            podium_count: podiumCount,
            avg_primary_metric_value: rawValues.length ? rawValues.reduce((acc, value) => acc + value, 0) / rawValues.length : null,
            primary_metric_name: primaryNames.length === 1 ? primaryNames[0] : "primary",
            avg_selection_metric_value: selectionValues.length ? selectionValues.reduce((acc, value) => acc + value, 0) / selectionValues.length : null,
            selection_metric_name: selectionNames.length === 1 ? selectionNames[0] : "",
          }};
        }})
        .sort((a, b) => a.avg_rank - b.avg_rank || b.win_count - a.win_count || b.podium_count - a.podium_count || b.datasets_used - a.datasets_used || a.model.localeCompare(b.model));
    }}

    function leaderboardColumns(rows) {{
      const present = sortByOrder(
        [...new Set(rows.map((row) => String(row.split_mode || "").toLowerCase()).filter(Boolean))],
        ["ws", "cs", "loso"]
      );
      let splits = ["ws", "cs"].filter((splitMode) => present.includes(splitMode));
      if (!splits.length) splits = present.slice(0, 2);
      if (splits.length === 1 && present.length > 1) {{
        const alternate = present.find((splitMode) => splitMode !== splits[0]);
        if (alternate) splits.push(alternate);
      }}
      if (splits.length === 1 && !["ws", "cs"].includes(splits[0])) splits.push("ws");
      if (splits.length === 1 && splits[0] === "ws") splits.push("cs");
      if (splits.length === 1 && splits[0] === "cs") splits.push("ws");
      if (!splits.length) splits = ["ws", "cs"];
      return splits.map((splitMode) => ({{
        split_mode: splitMode,
        label: splitLabel(splitMode),
        rows: overallRankingRows(rows.filter((row) => String(row.split_mode || "").toLowerCase() === splitMode)),
      }}));
    }}

    function selectedLeaderboardKey() {{
      return STATE.leaderboardSplitMode && STATE.leaderboardModel ? `${{STATE.leaderboardSplitMode}}::${{STATE.leaderboardModel}}` : "";
    }}

    function syncLeaderboardSelection(rows) {{
      const columns = leaderboardColumns(rows);
      const available = new Map();
      columns.forEach((column) => {{
        column.rows.forEach((row) => {{
          available.set(`${{column.split_mode}}::${{row.model}}`, {{ split_mode: column.split_mode, model: row.model }});
        }});
      }});
      if (!STATE.leaderboardModel) {{
        return columns;
      }}
      if (!available.has(selectedLeaderboardKey())) {{
        const sameModel = [...available.values()].find((entry) => entry.model === STATE.leaderboardModel) || null;
        if (sameModel) {{
          STATE.leaderboardSplitMode = sameModel.split_mode;
        }} else {{
          STATE.leaderboardSplitMode = "";
          STATE.leaderboardModel = "";
        }}
      }}
      return columns;
    }}

    function selectedLeaderboardEntry(rows) {{
      if (!STATE.leaderboardSplitMode || !STATE.leaderboardModel) return null;
      return overallRankingRows(rows.filter((row) => String(row.split_mode || "").toLowerCase() === STATE.leaderboardSplitMode)).find((entry) => entry.model === STATE.leaderboardModel) || null;
    }}

    function drilldownRows(rows) {{
      if (!STATE.leaderboardSplitMode || !STATE.leaderboardModel) return [];
      return bestByDataset(
        rows.filter((row) => row.is_rankable && String(row.split_mode || "").toLowerCase() === STATE.leaderboardSplitMode && row.model === STATE.leaderboardModel)
      );
    }}

    function syncDrilldownFocus(rows) {{
      const ids = new Set(rows.map((row) => row.effective_id));
      if (!ids.has(STATE.drilldownFocusEffectiveId)) {{
        STATE.drilldownFocusEffectiveId = rows[0] ? rows[0].effective_id : "";
      }}
      return rows.find((row) => row.effective_id === STATE.drilldownFocusEffectiveId) || rows[0] || null;
    }}

    function updateMatchingCounts() {{
      document.getElementById("matching-raw").textContent = filteredRawRows().length.toLocaleString("en-US");
      document.getElementById("matching-effective").textContent = filteredEffectiveRows().length.toLocaleString("en-US");
      document.getElementById("matching-rankable").textContent = filteredRankableRows().length.toLocaleString("en-US");
    }}

    function currentSelectionMetricLabel(rows) {{
      const selected = [...(STATE.filters.selection_metric_name || [])];
      if (selected.length === 1) return selected[0];
      const names = [...new Set(rows.map((row) => row.selection_metric_name).filter(Boolean))];
      return names.length === 1 ? names[0] : "selection";
    }}

    function filterOptionHtml(field, value, label, enabled, extraClass = "", titleText = "") {{
      return `<button
        class="filter-option ${{extraClass}}"
        type="button"
        data-filter-key="${{esc(field.key)}}"
        data-filter-value="${{esc(value)}}"
        title="${{esc(titleText || label)}}"
        ${{enabled ? "" : "disabled"}}
      >${{esc(label)}}</button>`;
    }}

    function useSelectFilter(field) {{
      const values = field.render_values || [];
      return ["suite_name", "model"].includes(field.key) || (field.key !== "dataset_key" && (values.length > 4 || values.some((value) => String(value).length > 14)));
    }}

    function renderCoreFilterField(field) {{
      const values = field.render_values || [];
      const selectMode = useSelectFilter(field);
      const note = field.enabled ? optionCountLabel(values.length) : (values.length ? optionCountLabel(values.length) : "Unavailable");
      if (field.key === "dataset_key") {{
        const summary = datasetFilterSummary(field);
        const options = [
          filterOptionHtml(field, "__any__", "Any dataset", true, "dropdown-option"),
          ...values.map((value) => filterOptionHtml(
            field,
            value,
            datasetFilterLabel(value),
            field.enabled,
            "dropdown-option",
            datasetFilterLabel(value),
          )),
        ].join("");
        return `
          <article class="compact-filter priority ${{field.enabled ? "" : "is-disabled"}}" data-core-filter="${{esc(field.key)}}">
            <div class="compact-filter-head">
              <label>${{esc(field.label)}}</label>
              <span class="compact-help">${{esc(note)}}</span>
            </div>
            <details class="multi-select-dropdown" data-multi-select="${{esc(field.key)}}">
              <summary class="multi-select-summary" data-multi-summary="${{esc(field.key)}}" title="${{esc(summary.title)}}">
                <span class="multi-select-summary-text">
                  <span class="multi-select-summary-value" data-multi-summary-value="${{esc(field.key)}}">${{esc(summary.value)}}</span>
                  <span class="multi-select-summary-meta" data-multi-summary-meta="${{esc(field.key)}}">${{esc(summary.meta)}}</span>
                </span>
                <span class="multi-select-caret" aria-hidden="true">▾</span>
              </summary>
              <div class="multi-select-panel">
                <div class="multi-select-scroll">${{options || "<span class='subtle'>No controls available.</span>"}}</div>
              </div>
            </details>
          </article>
        `;
      }}
      if (selectMode) {{
        const selectId = `filter-select-${{field.key}}`;
        const options = [
          `<option value="__any__">Any</option>`,
          ...values.map((value) => `<option value="${{esc(value)}}">${{esc(value)}}</option>`),
        ].join("");
        return `
          <article class="compact-filter priority ${{field.enabled ? "" : "is-disabled"}}" data-core-filter="${{esc(field.key)}}">
            <div class="compact-filter-head">
              <label for="${{esc(selectId)}}">${{esc(field.label)}}</label>
              <span class="compact-help">${{esc(note)}}</span>
            </div>
            <select id="${{esc(selectId)}}" class="compact-select" data-filter-select="${{esc(field.key)}}" ${{field.enabled ? "" : "disabled"}}>
              ${{options}}
            </select>
          </article>
        `;
      }}
      const options = [
        filterOptionHtml(field, "__any__", "Any", true, "compact"),
        ...values.map((value) => filterOptionHtml(
          field,
          value,
          field.key === "dataset_key" ? datasetFilterLabel(value) : value,
          field.enabled,
          "compact",
          datasetFilterLabel(value),
        )),
      ].join("");
      return `
        <article class="compact-filter priority ${{field.enabled ? "" : "is-disabled"}}" data-core-filter="${{esc(field.key)}}">
          <div class="compact-filter-head">
            <label>${{esc(field.label)}}</label>
            <span class="compact-help">${{esc(note)}}</span>
          </div>
          <div class="filter-inline-options ${{field.key === "dataset_key" ? "" : "compact-row"}}">${{options || "<span class='subtle'>No controls available.</span>"}}</div>
        </article>
      `;
    }}

    function renderFilterField(field) {{
      const metaLines = [];
      if (field.help_text) metaLines.push(field.help_text);
      if (field.default_value) metaLines.push(`Default: ${{field.default_value}}`);
      if (!field.enabled && field.disabled_reason) metaLines.push(field.disabled_reason);
      if (field.models && field.models.length) metaLines.push(`Models: ${{field.models.join(", ")}}`);
      const values = field.render_values || [];
      const options = [
        filterOptionHtml(field, "__any__", "Any", true),
        ...values.map((value) => filterOptionHtml(field, value, value, field.enabled)),
      ].join("");
      return `
        <article class="filter-card ${{field.enabled ? "" : "is-disabled"}}" data-filter-card="${{esc(field.key)}}">
          <div class="filter-title-row">
            <h4>${{esc(field.label)}}</h4>
            <span class="group-tag">${{esc(field.group)}}</span>
          </div>
          <div class="filter-options">${{options || "<span class='subtle'>No controls available.</span>"}}</div>
          <p class="filter-meta">${{esc(metaLines.join(" · "))}}</p>
        </article>
      `;
    }}

    function isModelParameterField(field) {{
      const modelCount = Array.isArray(APP.model_order) ? APP.model_order.length : 0;
      const explicitModelKeys = new Set(["variant", "patch_size", "patch_stride", "desired_time_len", "model_size", "depth"]);
      return !field.core && (
        explicitModelKeys.has(field.key) ||
        field.group === "Model Scoped" ||
        (Array.isArray(field.models) && field.models.length > 0 && (modelCount === 0 || field.models.length < modelCount))
      );
    }}

    function bindFilterControls() {{
      document.querySelectorAll("[data-filter-key]").forEach((button) => {{
        button.addEventListener("click", () => {{
          const key = button.dataset.filterKey;
          const value = button.dataset.filterValue;
          if (!STATE.filters[key]) STATE.filters[key] = new Set();
          if (value === "__any__") {{
            STATE.filters[key].clear();
          }} else {{
            if (STATE.filters[key].has(value)) {{
              STATE.filters[key].delete(value);
            }} else {{
              STATE.filters[key].add(value);
            }}
          }}
          refresh();
        }});
      }});
      document.querySelectorAll("[data-filter-select]").forEach((select) => {{
        select.addEventListener("change", () => {{
          const key = select.dataset.filterSelect;
          const value = select.value;
          STATE.filters[key] = value && value !== "__any__" ? new Set([value]) : new Set();
          refresh();
        }});
      }});
    }}

    function syncFilterControls() {{
      document.querySelectorAll("[data-filter-key]").forEach((button) => {{
        const key = button.dataset.filterKey;
        const value = button.dataset.filterValue;
        const selected = STATE.filters[key];
        const active = value === "__any__" ? !selected || selected.size === 0 : selected && selected.has(value);
        button.classList.toggle("active", Boolean(active));
        button.classList.toggle("any", value === "__any__");
      }});
      document.querySelectorAll("[data-filter-select]").forEach((select) => {{
        const key = select.dataset.filterSelect;
        const selected = STATE.filters[key];
        select.value = selected && selected.size ? [...selected][0] : "__any__";
      }});
      const datasetField = FILTER_FIELDS.find((field) => field.key === "dataset_key");
      if (datasetField) {{
        const summary = datasetFilterSummary(datasetField);
        document.querySelectorAll(`[data-multi-summary="dataset_key"]`).forEach((node) => {{
          node.title = summary.title;
        }});
        document.querySelectorAll(`[data-multi-summary-value="dataset_key"]`).forEach((node) => {{
          node.textContent = summary.value;
        }});
        document.querySelectorAll(`[data-multi-summary-meta="dataset_key"]`).forEach((node) => {{
          node.textContent = summary.meta;
        }});
      }}
    }}

    function renderFilters() {{
      const core = FILTER_FIELDS.filter((field) => field.core);
      const modelParams = FILTER_FIELDS.filter((field) => isModelParameterField(field));
      const advanced = FILTER_FIELDS.filter((field) => !field.core && !isModelParameterField(field));
      document.getElementById("core-filters").innerHTML = core.map(renderCoreFilterField).join("");
      document.getElementById("model-parameter-filters").innerHTML = modelParams.length
        ? modelParams.map(renderFilterField).join("")
        : "<p class='subtle'>No model-specific parameters were discovered for this result set.</p>";
      document.getElementById("advanced-filters").innerHTML = advanced.map(renderFilterField).join("");
      bindFilterControls();
      syncFilterControls();
    }}

    function renderEmptyChart(targetId, message) {{
      document.getElementById(targetId).innerHTML = `<div class="chart-empty">${{esc(message)}}</div>`;
    }}

    function renderLegend(targetId, items) {{
      const target = document.getElementById(targetId);
      if (!target) return;
      target.innerHTML = items.map((item) => `<span class="legend-pill"><span class="legend-swatch" style="background:${{esc(item.color)}}"></span>${{esc(item.label)}}</span>`).join("");
    }}

    function renderOverviewEmpty(targetId, message) {{
      const target = document.getElementById(targetId);
      if (!target) return;
      target.innerHTML = `<div class="overview-empty">${{esc(message)}}</div>`;
    }}

    function countRowsBy(rows, keyFn) {{
      const counts = new Map();
      rows.forEach((row) => {{
        const key = keyFn(row);
        counts.set(key, (counts.get(key) || 0) + 1);
      }});
      return counts;
    }}

    function orderedStatusKeys(counts) {{
      const extras = [...counts.keys()].filter((key) => !STATUS_ORDER.includes(key)).sort((a, b) => a.localeCompare(b));
      return [...STATUS_ORDER.filter((key) => (counts.get(key) || 0) > 0), ...extras];
    }}

    function renderOverviewStatusChart() {{
      const targetId = "overview-status";
      const counts = countRowsBy(RAW_ROWS, (row) => normalizeStatusBucket(row.status_bucket));
      const keys = orderedStatusKeys(counts);
      const total = [...counts.values()].reduce((acc, value) => acc + value, 0);
      if (!total) {{
        renderOverviewEmpty(targetId, "No raw runs are available for status aggregation.");
        renderLegend("overview-status-legend", []);
        return;
      }}
      const width = chartWidth(targetId, 360, 280, 420);
      const height = 236;
      const cx = width / 2;
      const cy = 102;
      const radius = 66;
      const strokeWidth = 24;
      const circumference = 2 * Math.PI * radius;
      let offset = 0;
      const svg = [responsiveSvgFrame(width, height, "Run status overview")];
      svg.push(`<circle cx="${{cx}}" cy="${{cy}}" r="${{radius}}" fill="none" stroke="rgba(255,255,255,0.13)" stroke-width="${{strokeWidth}}"></circle>`);
      keys.forEach((key) => {{
        const count = counts.get(key) || 0;
        const length = (count / total) * circumference;
        svg.push(
          `<circle cx="${{cx}}" cy="${{cy}}" r="${{radius}}" fill="none" stroke="${{statusColor(key)}}" stroke-width="${{strokeWidth}}" stroke-linecap="butt" stroke-dasharray="${{length}} ${{circumference - length}}" stroke-dashoffset="${{-offset}}" transform="rotate(-90 ${{cx}} ${{cy}})"></circle>`
        );
        offset += length;
      }});
      svg.push(`<circle cx="${{cx}}" cy="${{cy}}" r="46" fill="rgba(11, 34, 38, 0.28)"></circle>`);
      svg.push(`<text x="${{cx}}" y="${{cy - 4}}" text-anchor="middle" fill="#ffffff" font-size="28" font-weight="700">${{total.toLocaleString("en-US")}}</text>`);
      svg.push(`<text x="${{cx}}" y="${{cy + 18}}" text-anchor="middle" fill="rgba(255,255,255,0.82)" font-size="11" letter-spacing="1.4">RAW RUNS</text>`);
      svg.push("</svg>");
      document.getElementById(targetId).innerHTML = svg.join("");
      renderLegend(
        "overview-status-legend",
        keys.map((key) => ({{
          label: `${{statusLabel(key)}} · ${{(counts.get(key) || 0).toLocaleString("en-US")}}`,
          color: statusColor(key),
        }}))
      );
    }}

    function renderOverviewSplitChart() {{
      const targetId = "overview-split";
      const counts = countRowsBy(RAW_ROWS, (row) => normalizeGroupLabel(row.split_mode));
      const categories = sortByOrder([...counts.keys()], ["ws", "cs", "loso", "Unknown"]);
      const total = [...counts.values()].reduce((acc, value) => acc + value, 0);
      const note = document.getElementById("overview-split-note");
      if (!categories.length) {{
        renderOverviewEmpty(targetId, "No raw runs are available for split-mode aggregation.");
        if (note) note.textContent = "";
        return;
      }}
      const width = chartWidth(targetId, 360, 280, 440);
      const height = 236;
      const margin = {{ top: 16, right: 16, bottom: 48, left: 38 }};
      const plotWidth = width - margin.left - margin.right;
      const plotHeight = height - margin.top - margin.bottom;
      const maxValue = Math.max(...categories.map((category) => counts.get(category) || 0), 1);
      const slot = plotWidth / Math.max(categories.length, 1);
      const barWidth = Math.min(72, slot * 0.56);
      const x = (index) => margin.left + slot * index + (slot - barWidth) / 2;
      const y = (value) => margin.top + (1 - value / maxValue) * plotHeight;
      const svg = [responsiveSvgFrame(width, height, "Runs by split mode overview")];
      for (let tickIndex = 0; tickIndex <= 4; tickIndex += 1) {{
        const raw = (maxValue * tickIndex) / 4;
        const py = y(raw);
        svg.push(`<line x1="${{margin.left}}" y1="${{py}}" x2="${{width - margin.right}}" y2="${{py}}" stroke="rgba(255,255,255,0.14)"></line>`);
        svg.push(`<text x="${{margin.left - 8}}" y="${{py + 4}}" text-anchor="end" fill="rgba(255,255,255,0.78)" font-size="10">${{Math.round(raw)}}</text>`);
      }}
      categories.forEach((category, index) => {{
        const value = counts.get(category) || 0;
        const px = x(index);
        const py = y(value);
        const color = category === "Unknown" ? statusColor("unknown") : splitColor(category);
        const label = category === "Unknown" ? category : category.toUpperCase();
        svg.push(`<rect x="${{px}}" y="${{py}}" width="${{barWidth}}" height="${{Math.max(0, margin.top + plotHeight - py)}}"
          rx="10" fill="${{color}}" opacity="0.94"></rect>`);
        svg.push(`<text x="${{px + barWidth / 2}}" y="${{Math.max(12, py - 6)}}" text-anchor="middle" fill="#ffffff" font-size="11" font-weight="700">${{value}}</text>`);
        svg.push(`<text x="${{px + barWidth / 2}}" y="${{height - 16}}" text-anchor="middle" fill="rgba(255,255,255,0.84)" font-size="11">${{esc(shortLabel(label, 10))}}</text>`);
      }});
      svg.push("</svg>");
      document.getElementById(targetId).innerHTML = svg.join("");
      if (note) {{
        const lead = categories
          .map((category) => [category, counts.get(category) || 0])
          .sort((left, right) => right[1] - left[1] || String(left[0]).localeCompare(String(right[0])))[0];
        const leadLabel = lead ? (lead[0] === "Unknown" ? "Unknown" : String(lead[0]).toUpperCase()) : "Unknown";
        note.textContent = `${{categories.length}} split mode(s) across ${{total.toLocaleString("en-US")}} raw runs. Largest slice: ${{leadLabel}} (${{lead ? lead[1] : 0}}).`;
      }}
    }}

    function renderOverviewSuiteChart() {{
      const targetId = "overview-suite";
      const suiteMap = new Map();
      RAW_ROWS.forEach((row) => {{
        const suite = normalizeGroupLabel(row.suite_name);
        const status = normalizeStatusBucket(row.status_bucket);
        const entry = suiteMap.get(suite) || {{ total: 0, statuses: new Map() }};
        entry.total += 1;
        entry.statuses.set(status, (entry.statuses.get(status) || 0) + 1);
        suiteMap.set(suite, entry);
      }});
      const suites = sortByOrder([...suiteMap.keys()], (APP.summary && Array.isArray(APP.summary.suite_names) ? APP.summary.suite_names : []).map(normalizeGroupLabel));
      if (!suites.length) {{
        renderOverviewEmpty(targetId, "No raw runs are available for suite aggregation.");
        renderLegend("overview-suite-legend", []);
        return;
      }}
      const overallStatus = countRowsBy(RAW_ROWS, (row) => normalizeStatusBucket(row.status_bucket));
      const statusKeys = orderedStatusKeys(overallStatus);
      const maxTotal = Math.max(...suites.map((suite) => suiteMap.get(suite)?.total || 0), 1);
      const width = chartWidth(targetId, 460, 320, 620);
      const rowHeight = 36;
      const height = Math.max(196, 56 + suites.length * rowHeight);
      const margin = {{ top: 16, right: 44, bottom: 18, left: 108 }};
      const plotWidth = width - margin.left - margin.right;
      const svg = [responsiveSvgFrame(width, height, "Runs by suite overview")];
      suites.forEach((suite, index) => {{
        const entry = suiteMap.get(suite);
        const y = margin.top + index * rowHeight + 8;
        const barHeight = 18;
        const totalWidth = (entry.total / maxTotal) * plotWidth;
        let cursor = margin.left;
        svg.push(`<text x="${{margin.left - 10}}" y="${{y + 13}}" text-anchor="end" fill="#ffffff" font-size="11">${{esc(shortLabel(suite, 18))}}</text>`);
        svg.push(`<rect x="${{margin.left}}" y="${{y}}" width="${{plotWidth}}" height="${{barHeight}}" rx="9" fill="rgba(255,255,255,0.1)"></rect>`);
        statusKeys.forEach((status) => {{
          const count = entry.statuses.get(status) || 0;
          if (!count) return;
          const segmentWidth = (count / maxTotal) * plotWidth;
          svg.push(`<rect x="${{cursor}}" y="${{y}}" width="${{segmentWidth}}" height="${{barHeight}}" rx="9" fill="${{statusColor(status)}}"></rect>`);
          cursor += segmentWidth;
        }});
        svg.push(`<text x="${{margin.left + totalWidth + 8}}" y="${{y + 13}}" fill="rgba(255,255,255,0.86)" font-size="11" font-weight="700">${{entry.total}}</text>`);
      }});
      svg.push(`<text x="${{margin.left}}" y="12" fill="rgba(255,255,255,0.72)" font-size="10">0</text>`);
      svg.push(`<text x="${{width - margin.right}}" y="12" text-anchor="end" fill="rgba(255,255,255,0.72)" font-size="10">${{maxTotal}}</text>`);
      svg.push("</svg>");
      document.getElementById(targetId).innerHTML = svg.join("");
      renderLegend(
        "overview-suite-legend",
        statusKeys.map((status) => ({{
          label: `${{statusLabel(status)}} · ${{(overallStatus.get(status) || 0).toLocaleString("en-US")}}`,
          color: statusColor(status),
        }}))
      );
    }}

    function renderOverviewCharts() {{
      renderOverviewStatusChart();
      renderOverviewSplitChart();
      renderOverviewSuiteChart();
    }}

    function heatmapPages(models) {{
      const pages = [];
      for (let index = 0; index < models.length; index += HEATMAP_MODELS_PER_PAGE) {{
        pages.push(models.slice(index, index + HEATMAP_MODELS_PER_PAGE));
      }}
      return pages;
    }}

    function bindHeatmapControls() {{
      document.querySelectorAll("[data-heatmap-page]").forEach((node) => {{
        node.addEventListener("click", () => {{
          const page = Number.parseInt(node.dataset.heatmapPage || "0", 10);
          if (!Number.isFinite(page)) return;
          STATE.heatmapPage = page;
          renderHeatmap(filteredRankableRows());
        }});
      }});
    }}

    function renderHeatmapPagination(models, pages) {{
      const target = document.getElementById("heatmap-pagination");
      if (!target) return;
      if (!models.length || pages.length <= 1) {{
        target.innerHTML = "";
        return;
      }}
      const safePage = clamp(STATE.heatmapPage, 0, pages.length - 1);
      const start = safePage * HEATMAP_MODELS_PER_PAGE + 1;
      const end = Math.min(models.length, start + pages[safePage].length - 1);
      const buttons = pages.map((pageModels, index) => {{
        const pageStart = index * HEATMAP_MODELS_PER_PAGE + 1;
        const pageEnd = Math.min(models.length, pageStart + pageModels.length - 1);
        return `<button class="ghost-button page-button ${{index === safePage ? "active" : ""}}" type="button" data-heatmap-page="${{index}}">${{pageStart}}-${{pageEnd}}</button>`;
      }}).join("");
      target.innerHTML = `<span class="subtle">Models ${{start}}-${{end}} of ${{models.length}}</span>${{buttons}}`;
      bindHeatmapControls();
    }}

    function splitDatasetCount(rows, splitMode) {{
      return new Set(
        rows
          .filter((row) => String(row.split_mode || "").toLowerCase() === splitMode)
          .map((row) => row.dataset_key)
          .filter(Boolean)
      ).size;
    }}

    function leaderboardMetricLabel(row, datasetCount) {{
      const base = row.primary_metric_name || "primary";
      return datasetCount <= 1 ? base : `mean ${{base}}`;
    }}

    function leaderboardRankLabel(index, row, datasetCount) {{
      return datasetCount <= 1 ? `rank ${{index + 1}}` : `avg rank ${{fmt(row.avg_rank, 2)}}`;
    }}

    function leaderboardColumnNote(column, datasetCount) {{
      return datasetCount <= 1
        ? `Primary-score ranking within ${{column.label}} only`
        : `Average dataset rank within ${{column.label}} across ${{datasetCount}} dataset${{datasetCount === 1 ? "" : "s"}}`;
    }}

    function overallRankingNoteText(top, rows) {{
      if (!top) {{
        return "Widen the filters to recover rankable runs and the paired split curves.";
      }}
      const datasetCount = new Set(rows.map((row) => row.dataset_key).filter(Boolean)).size;
      const rankingCopy = datasetCount <= 1
        ? "Current filters isolate a single dataset, so bars are ordered by that dataset's primary score."
        : `Current filters span ${{datasetCount}} datasets, so bars are ordered by average dataset rank while the shown metric stays a mean primary score.`;
      return `${{rankingCopy}} Hover a bar or curve to isolate the same model inside its split column. Click once to pin the same model across WS/CS, then click the pinned model again to release it before hovering or pinning another model. ${{top.model}} currently leads one of the filtered split-specific boards.`;
    }}

    function renderOverallRanking(rows, columns = leaderboardColumns(rows)) {{
      const target = document.getElementById("overall-ranking");
      const hasAnyRows = columns.some((column) => column.rows.length);
      if (!hasAnyRows) {{
        target.innerHTML = `
          <div class="leaderboard-column" style="grid-column:1 / -1;">
            <div class="leaderboard-empty">No rankable conditions match the current filters.</div>
          </div>
        `;
        document.getElementById("overall-ranking-note").textContent = "Try widening the filters or re-enable completed runs.";
        return;
      }}

      target.innerHTML = columns.map((column) => {{
        const displayed = column.rows.slice(0, 12);
        const datasetCount = splitDatasetCount(rows, column.split_mode);
        const accent = splitColor(column.split_mode);
        const bars = displayed.length
          ? displayed.map((row, index) => {{
              const fill = displayed.length <= 1
                ? 1
                : clamp(1 - (index / Math.max(displayed.length - 1, 1)) * 0.82, 0.18, 1);
              const barColor = mixColor(modelColor(row.model), accent, 0.36);
              const barSoft = mixColor(barColor, "#ffffff", 0.76);
              const barDeep = mixColor(barColor, "#18232c", 0.16);
              const metricLabel = leaderboardMetricLabel(row, datasetCount);
              const rankLabel = leaderboardRankLabel(index, row, datasetCount);
              return `
                <button
                  class="leaderboard-bar"
                  type="button"
                  data-highlight-key="${{esc(row.model)}}"
                  data-highlight-split="${{esc(column.split_mode)}}"
                  style="--bar-fill:${{Math.round(fill * 100)}}%;--bar-color:${{barColor}};--bar-soft:${{barSoft}};--bar-deep:${{barDeep}};"
                >
                  <span class="leaderboard-rank-badge">${{String(index + 1).padStart(2, "0")}}</span>
                  <span class="leaderboard-content">
                    <span class="leaderboard-topline">
                      <span class="leaderboard-model">${{esc(row.model)}}</span>
                      <span class="leaderboard-stats">
                        <span class="leaderboard-stat">${{esc(rankLabel)}}</span>
                        <span class="leaderboard-stat metric">${{esc(metricLabel)}} ${{fmt(row.avg_primary_metric_value)}}</span>
                        <span class="leaderboard-stat">wins ${{row.win_count}}</span>
                        <span class="leaderboard-stat">podium ${{row.podium_count}}</span>
                        <span class="leaderboard-stat">${{row.datasets_used}} dataset${{row.datasets_used === 1 ? "" : "s"}}</span>
                      </span>
                    </span>
                  </span>
                </button>
              `;
            }}).join("")
          : `<div class="leaderboard-empty">No matching rankable runs for ${{column.label}} under the current filters.</div>`;
        return `
          <section class="leaderboard-column">
            <div class="leaderboard-column-head">
              <div>
                <h3 style="color:${{accent}}">${{esc(column.label)}}</h3>
                <p>${{displayed.length}} shown · ${{column.rows.length}} ranked model(s)</p>
              </div>
              <p>${{esc(leaderboardColumnNote(column, datasetCount))}}</p>
            </div>
            <div class="leaderboard-stack">${{bars}}</div>
            <div class="leaderboard-column-curve">
              <div id="split-curve-${{esc(column.split_mode)}}" class="chart-shell split-curve-shell"></div>
              <div id="split-curve-legend-${{esc(column.split_mode)}}" class="legend-row"></div>
            </div>
          </section>
        `;
      }}).join("");
      const top = columns.find((column) => column.rows.length)?.rows[0] || null;
      document.getElementById("overall-ranking-note").textContent = overallRankingNoteText(top, rows);
    }}

    function aggregateCurveSeries(rows, mode) {{
      const epochMap = new Map();
      rows.forEach((row) => {{
        if (!Array.isArray(row.curve_series)) return;
        row.curve_series.forEach((point) => {{
          const value = num(point[mode]);
          const epoch = num(point.epoch);
          if (value === null || epoch === null) return;
          const entry = epochMap.get(epoch) || {{ sum: 0, count: 0 }};
          entry.sum += value;
          entry.count += 1;
          epochMap.set(epoch, entry);
        }});
      }});
      return [...epochMap.entries()]
        .map(([epoch, entry]) => ({{
          epoch,
          value: entry.count ? entry.sum / entry.count : null,
        }}))
        .filter((point) => point.value !== null)
        .sort((left, right) => left.epoch - right.epoch);
    }}

    function splitCurveRows(rows, rankingRows, splitMode) {{
      return rankingRows.slice(0, 12).map((entry) => {{
        const modelRows = rows.filter((row) => row.is_rankable && String(row.split_mode || "").toLowerCase() === splitMode && row.model === entry.model);
        return {{
          model: entry.model,
          split_mode: splitMode,
          avg_rank: entry.avg_rank,
          avg_selection_metric_value: entry.avg_selection_metric_value,
          selection_metric_name: entry.selection_metric_name,
          series: aggregateCurveSeries(modelRows, STATE.explorerCurveMode),
        }};
      }}).filter((item) => item.series.length);
    }}

    function applyHoverState() {{
      const hasHover = Boolean(STATE.hoverSplitMode && STATE.hoverModel);
      const hasSelection = Boolean(STATE.leaderboardModel);
      document.querySelectorAll("[data-highlight-key]").forEach((node) => {{
        const sameSplit = node.dataset.highlightSplit === STATE.hoverSplitMode;
        const sameKey = node.dataset.highlightKey === STATE.hoverModel;
        const selectedKey = node.dataset.highlightKey === STATE.leaderboardModel;
        const selectedSplit = node.dataset.highlightSplit === STATE.leaderboardSplitMode;
        const selectionSelected = hasSelection && selectedKey && selectedSplit;
        const selectionLinked = hasSelection && selectedKey && !selectedSplit;
        const selectionMuted = hasSelection && !selectedKey;
        const hoverMatched = hasHover && sameSplit && sameKey;
        const hoverMuted = hasHover && sameSplit && !sameKey;
        node.classList.toggle("is-selected", selectionSelected);
        node.classList.toggle("is-linked", selectionLinked);
        node.classList.toggle("is-hovered", hoverMatched);
        node.classList.toggle("is-muted", hasHover ? (hoverMuted || (selectionMuted && !sameKey)) : selectionMuted);
      }});
    }}

    function bindHighlightInteractions() {{
      document.querySelectorAll("[data-highlight-key]").forEach((node) => {{
        node.addEventListener("click", () => {{
          const split = node.dataset.highlightSplit || "";
          const model = node.dataset.highlightKey || "";
          if (!model) return;
          if (STATE.leaderboardModel) {{
            if (STATE.leaderboardModel === model) {{
              STATE.leaderboardSplitMode = "";
              STATE.leaderboardModel = "";
              STATE.hoverSplitMode = "";
              STATE.hoverModel = "";
              applyHoverState();
            }}
            return;
          }}
          if (STATE.leaderboardSplitMode === split && STATE.leaderboardModel === model) {{
            STATE.leaderboardSplitMode = "";
            STATE.leaderboardModel = "";
          }} else {{
            STATE.leaderboardSplitMode = split;
            STATE.leaderboardModel = model;
          }}
          STATE.hoverSplitMode = "";
          STATE.hoverModel = "";
          applyHoverState();
        }});
        node.addEventListener("mouseenter", () => {{
          if (STATE.leaderboardModel) return;
          STATE.hoverSplitMode = node.dataset.highlightSplit || "";
          STATE.hoverModel = node.dataset.highlightKey || "";
          applyHoverState();
        }});
        node.addEventListener("mouseleave", () => {{
          if (STATE.leaderboardModel) return;
          if (STATE.hoverSplitMode === (node.dataset.highlightSplit || "") && STATE.hoverModel === (node.dataset.highlightKey || "")) {{
            STATE.hoverSplitMode = "";
            STATE.hoverModel = "";
            applyHoverState();
          }}
        }});
      }});
      applyHoverState();
    }}

    function renderSingleSplitCurve(rows, column) {{
      const targetId = `split-curve-${{column.split_mode}}`;
      const legendId = `split-curve-legend-${{column.split_mode}}`;
      const seriesRows = splitCurveRows(rows, column.rows, column.split_mode);
      if (!seriesRows.length) {{
        renderEmptyChart(targetId, `No curve series are available for ${{column.label}} under the current filters.`);
        renderLegend(legendId, []);
        return;
      }}
      const width = chartWidth(targetId, 560, 360, 760);
      const height = 276;
      const margin = {{ top: 38, right: 16, bottom: 46, left: 52 }};
      const plotWidth = width - margin.left - margin.right;
      const plotHeight = height - margin.top - margin.bottom;
      const mode = STATE.explorerCurveMode;
      const maxEpoch = Math.max(...seriesRows.flatMap((item) => item.series.map((point) => point.epoch)), 1);
      const values = seriesRows.flatMap((item) => item.series.map((point) => point.value));
      const bounds = curveAxisBounds(values, mode);
      const x = (epoch) => maxEpoch <= 1 ? margin.left + plotWidth / 2 : margin.left + ((epoch - 1) / (maxEpoch - 1)) * plotWidth;
      const y = (value) => margin.top + (1 - ((value - bounds.minValue) / ((bounds.maxValue - bounds.minValue) || 1))) * plotHeight;
      const svg = [
        responsiveSvgFrame(width, height, `${{column.label}} curves`),
        `<text x="${{margin.left}}" y="20" fill="#172026" font-size="13" font-weight="700">${{esc(column.label)}} · ${{esc(mode.replace("_", " "))}}</text>`,
        `<text x="${{margin.left}}" y="34" fill="#5c666d" font-size="10.5">Each line averages the filtered curve values for one leaderboard model within this split.</text>`,
      ];
      for (let tickIndex = 0; tickIndex <= 4; tickIndex += 1) {{
        const raw = bounds.minValue + ((bounds.maxValue - bounds.minValue) * tickIndex) / 4;
        const py = y(raw);
        svg.push(`<line x1="${{margin.left}}" y1="${{py}}" x2="${{width - margin.right}}" y2="${{py}}" stroke="#e5e2d9"></line>`);
        svg.push(`<text x="${{margin.left - 8}}" y="${{py + 4}}" text-anchor="end" fill="#5c666d" font-size="11">${{fmt(raw, mode === "lr" ? 5 : 3)}}</text>`);
      }}
      const epochTicks = [...new Set([1, Math.max(1, Math.round(maxEpoch / 2)), maxEpoch])].sort((a, b) => a - b);
      epochTicks.forEach((epoch) => {{
        const px = x(epoch);
        svg.push(`<line x1="${{px}}" y1="${{margin.top}}" x2="${{px}}" y2="${{height - margin.bottom + 4}}" stroke="#ece7dc"></line>`);
        svg.push(`<text x="${{px}}" y="${{height - 16}}" text-anchor="middle" fill="#5c666d" font-size="11">ep ${{epoch}}</text>`);
      }});
      const legendItems = [];
      seriesRows.forEach((item) => {{
        const color = modelColor(item.model);
        const points = item.series.map((point) => `${{x(point.epoch)}},${{y(point.value)}}`);
        legendItems.push({{
          label: `${{item.model}} · avg rank ${{fmt(item.avg_rank, 2)}}`,
          color,
        }});
        svg.push(`<g class="split-curve-series" data-highlight-key="${{esc(item.model)}}" data-highlight-split="${{esc(column.split_mode)}}" style="cursor:pointer">`);
        svg.push(`<polyline data-curve-underlay fill="none" stroke="#ffffff" stroke-width="7.5" stroke-linecap="round" stroke-linejoin="round" opacity="0" points="${{points.join(" ")}}"></polyline>`);
        svg.push(`<polyline data-curve-main fill="none" stroke="${{color}}" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round" opacity="0.78" points="${{points.join(" ")}}"></polyline>`);
        item.series.forEach((point, index) => {{
          const isLast = index === item.series.length - 1;
          svg.push(`<circle data-curve-point cx="${{x(point.epoch)}}" cy="${{y(point.value)}}" r="${{isLast ? 3.2 : 2.4}}" fill="${{color}}" opacity="0.76"></circle>`);
        }});
        svg.push("</g>");
      }});
      svg.push("</svg>");
      document.getElementById(targetId).innerHTML = svg.join("");
      renderLegend(legendId, legendItems);
    }}

    function renderSplitCurves(rows, columns) {{
      columns.forEach((column) => renderSingleSplitCurve(rows, column));
      bindHighlightInteractions();
    }}

    function renderHeatmap(rows) {{
      const rankable = rows.filter((row) => row.is_rankable);
      if (!rankable.length) {{
        renderEmptyChart("heatmap", "No rankable cells are available for the current filters.");
        renderHeatmapPagination([], []);
        return;
      }}
      const datasets = sortByOrder(
        [...new Set(rankable.map((row) => row.dataset_key))],
        APP.dataset_order
      );
      const allModels = sortByOrder(
        [...new Set(rankable.map((row) => row.model))],
        APP.model_order
      );
      const pages = heatmapPages(allModels);
      STATE.heatmapPage = clamp(STATE.heatmapPage || 0, 0, Math.max(0, pages.length - 1));
      const models = pages[STATE.heatmapPage] || allModels;
      renderHeatmapPagination(allModels, pages);
      const intensities = datasetRankIntensityMap(rankable);
      const bestCell = new Map();
      rankable.forEach((row) => {{
        const key = `${{row.dataset_key}}::${{row.model}}`;
        const current = bestCell.get(key);
        if (!current || compareRows(row, current) < 0) {{
          bestCell.set(key, row);
        }}
      }});
      const targetId = "heatmap";
      const availableWidth = chartWidth(targetId, 1180, 760, 1320);
      const leftLabelWidth = clamp(Math.round(availableWidth * 0.22), 188, 248);
      const preferredCellWidth = Math.floor((availableWidth - leftLabelWidth - 24) / Math.max(models.length, 1));
      const useOverflow = preferredCellWidth < 60;
      const cellWidth = clamp(preferredCellWidth, 60, 160);
      const cellHeight = clamp(52 - Math.max(0, models.length - 8), 44, 52);
      const cellBoxWidth = Math.max(44, cellWidth - 10);
      const cellBoxHeight = Math.max(34, cellHeight - 10);
      const width = useOverflow ? leftLabelWidth + 24 + models.length * cellWidth : availableWidth;
      const height = 104 + datasets.length * cellHeight;
      const modelLabelFontSize = cellWidth < 72 ? 10 : 11;
      const modelLabelLimit = cellWidth < 72 ? 10 : 15;
      const rowLabelFontSize = leftLabelWidth < 210 ? 11 : 12;
      const rowLabelLimit = leftLabelWidth < 210 ? 30 : 36;
      const metricDigits = cellWidth < 72 ? 2 : 3;
      const svg = [];
      svg.push((useOverflow ? svgFrame : responsiveSvgFrame)(width, height, "Model dataset heatmap"));
      svg.push(`<text x="18" y="24" fill="#172026" font-size="14" font-weight="700">Best filtered run per model and dataset</text>`);
      models.forEach((model, index) => {{
        const x = leftLabelWidth + index * cellWidth + cellWidth / 2;
        svg.push(`<text x="${{x}}" y="68" fill="${{modelColor(model)}}" font-size="${{modelLabelFontSize}}" font-weight="700" text-anchor="middle">${{esc(shortLabel(model, modelLabelLimit))}}</text>`);
      }});
      datasets.forEach((datasetKey, rowIndex) => {{
        const y = 88 + rowIndex * cellHeight;
        svg.push(`<text x="${{leftLabelWidth - 12}}" y="${{y + cellBoxHeight / 2 + 4}}" fill="#172026" font-size="${{rowLabelFontSize}}" text-anchor="end">${{esc(shortLabel(datasetKey.replace(":", " / "), rowLabelLimit))}}</text>`);
        models.forEach((model, columnIndex) => {{
          const key = `${{datasetKey}}::${{model}}`;
          const row = bestCell.get(key);
          const x = leftLabelWidth + columnIndex * cellWidth;
          if (!row) {{
            svg.push(`<rect x="${{x}}" y="${{y}}" width="${{cellBoxWidth}}" height="${{cellBoxHeight}}" rx="10" fill="#f1eee7" stroke="#dedad1"></rect>`);
            svg.push(`<text x="${{x + cellBoxWidth / 2}}" y="${{y + cellBoxHeight / 2 + 4}}" fill="#9a968d" font-size="11" text-anchor="middle">-</text>`);
            return;
          }}
          const fill = heatTone(intensities.get(row.effective_id));
          svg.push(`<rect x="${{x}}" y="${{y}}" width="${{cellBoxWidth}}" height="${{cellBoxHeight}}" rx="10" fill="${{fill}}" stroke="#d5d7cf"></rect>`);
          svg.push(`<text x="${{x + cellBoxWidth / 2}}" y="${{y + cellBoxHeight / 2 + 4}}" fill="#172026" font-size="11" font-weight="600" text-anchor="middle">${{fmt(row.primary_metric_value, metricDigits)}}</text>`);
        }});
      }});
      svg.push("</svg>");
      document.getElementById("heatmap").innerHTML = svg.join("");
    }}

    function compareSeries(rows, key) {{
      const rankable = rows.filter((row) => row.is_rankable);
      if (!rankable.length) return [];
      const buckets = new Map();
      rankable.forEach((row) => {{
        const datasetKey = row.dataset_key;
        const valueKey = String(row[key] ?? "");
        if (!datasetKey || !valueKey) return;
        const metric = num(row.primary_metric_value);
        if (metric === null || metric === undefined) return;
        if (!buckets.has(datasetKey)) buckets.set(datasetKey, new Map());
        if (!buckets.get(datasetKey).has(valueKey)) buckets.get(datasetKey).set(valueKey, []);
        buckets.get(datasetKey).get(valueKey).push(metric);
      }});
      return [...buckets.entries()].map(([datasetKey, valueMap]) => {{
        const points = [...valueMap.entries()].map(([valueKey, values]) => ({{
          key: valueKey,
          value: values.reduce((acc, item) => acc + item, 0) / values.length,
        }}));
        return {{
          datasetKey,
          color: datasetColor(datasetKey),
          points,
        }};
      }});
    }}

    function renderLineCompare(targetId, legendId, title, key, order, rows) {{
      const series = compareSeries(rows, key);
      if (!series.length) {{
        renderEmptyChart(targetId, "No completed runs are available for this comparison.");
        renderLegend(legendId, []);
        return;
      }}
      const width = 520;
      const height = 332;
      const margin = {{ top: 44, right: 18, bottom: 62, left: 50 }};
      const plotWidth = width - margin.left - margin.right;
      const plotHeight = height - margin.top - margin.bottom;
      const categories = sortByOrder(order.length ? order : [...new Set(series.flatMap((item) => item.points.map((point) => point.key)))], order);
      const values = series.flatMap((item) => item.points.map((point) => point.value));
      const minValue = Math.min(...values);
      const maxValue = Math.max(...values);
      const span = Math.max(0.02, maxValue - minValue);
      const bounded = values.every((value) => value >= 0 && value <= 1);
      const padding = span * 0.18;
      const yMin = bounded ? Math.max(0, minValue - padding) : minValue - padding;
      const yMax = bounded ? Math.min(1, maxValue + padding) : maxValue + padding;
      const x = (index) => categories.length <= 1 ? margin.left + plotWidth / 2 : margin.left + (plotWidth * index) / (categories.length - 1);
      const y = (value) => margin.top + (1 - ((value - yMin) / ((yMax - yMin) || 1))) * plotHeight;
      const legendItems = [];
      const pieces = [
        svgFrame(width, height, title),
        `<text x="${{margin.left}}" y="20" fill="#172026" font-size="13" font-weight="700">${{esc(title)}}</text>`,
        `<text x="${{margin.left}}" y="36" fill="#5c666d" font-size="11">Mean raw primary metric within the current filtered subset</text>`,
      ];
      for (let tickIndex = 0; tickIndex <= 4; tickIndex += 1) {{
        const raw = yMin + ((yMax - yMin) * tickIndex) / 4;
        const py = y(raw);
        pieces.push(`<line x1="${{margin.left}}" y1="${{py}}" x2="${{width - margin.right}}" y2="${{py}}" stroke="#e5e2d9"></line>`);
        pieces.push(`<text x="${{margin.left - 8}}" y="${{py + 4}}" text-anchor="end" fill="#5c666d" font-size="11">${{fmt(raw, 3)}}</text>`);
      }}
      categories.forEach((category, index) => {{
        const px = x(index);
        pieces.push(`<text x="${{px}}" y="${{height - 20}}" text-anchor="middle" fill="#5c666d" font-size="11">${{esc(shortLabel(category, 12))}}</text>`);
      }});
      series.forEach((item) => {{
        legendItems.push({{ label: item.datasetKey, color: item.color }});
        const pointMap = new Map(item.points.map((point) => [point.key, point.value]));
        const points = categories
          .map((category, index) => pointMap.has(category) ? `${{x(index)}},${{y(pointMap.get(category))}}` : "")
          .filter(Boolean);
        if (points.length >= 2) {{
          pieces.push(`<polyline fill="none" stroke="${{item.color}}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" points="${{points.join(" ")}}"></polyline>`);
        }}
        categories.forEach((category, index) => {{
          if (!pointMap.has(category)) return;
          pieces.push(`<circle cx="${{x(index)}}" cy="${{y(pointMap.get(category))}}" r="4" fill="${{item.color}}"></circle>`);
        }});
      }});
      pieces.push("</svg>");
      document.getElementById(targetId).innerHTML = pieces.join("");
      renderLegend(legendId, legendItems);
    }}

    function renderComparisons(rows) {{
      renderLineCompare("compare-split", "compare-split-legend", "Split Mode", "split_mode", ["cs", "ws", "loso"], rows);
      renderLineCompare("compare-channel", "compare-channel-legend", "Channel System", "channel_system", ["10-20", "10-10", "10-5", "dynamic", "custom"], rows);
      renderLineCompare("compare-clr", "compare-clr-legend", "Classifier LR Scale", "classifier_lr_scale", ["1", "5", "10"], rows);
    }}

    function curveAxisBounds(values, mode) {{
      let minValue = Math.min(...values);
      let maxValue = Math.max(...values);
      if (minValue === maxValue) {{
        const baseline = Math.abs(maxValue) > 1e-9 ? Math.abs(maxValue) : (mode === "lr" ? 1e-4 : 0.05);
        const padding = baseline * (mode === "lr" ? 0.22 : 0.14);
        minValue -= padding;
        maxValue += padding;
      }} else {{
        const padding = (maxValue - minValue) * (mode === "lr" ? 0.12 : 0.14);
        minValue -= padding;
        maxValue += padding;
      }}
      const bounded = mode !== "lr" && values.every((value) => value >= 0 && value <= 1.05);
      if (bounded) {{
        minValue = clamp(minValue, 0, 1);
        maxValue = clamp(maxValue, 0, 1);
        if (maxValue - minValue < 0.06) {{
          const midpoint = (maxValue + minValue) / 2;
          minValue = clamp(midpoint - 0.03, 0, 1);
          maxValue = clamp(midpoint + 0.03, 0, 1);
        }}
      }}
      return {{ minValue, maxValue }};
    }}

    function bindDrilldownFocusControls() {{
      document.querySelectorAll("[data-drilldown-focus]").forEach((node) => {{
        node.addEventListener("click", () => {{
          STATE.drilldownFocusEffectiveId = node.dataset.drilldownFocus;
          renderExplorer(filteredRankableRows());
        }});
      }});
    }}

    function renderDrilldownDatasetCards(rows, focusRow) {{
      const target = document.getElementById("drilldown-datasets");
      if (!rows.length) {{
        target.innerHTML = `<div class="subtle">No dataset-level breakdown is available for the current leaderboard selection.</div>`;
        return;
      }}
      target.innerHTML = rows.map((row) => {{
        const color = datasetColor(row.dataset_key);
        const selected = focusRow && row.effective_id === focusRow.effective_id;
        return `
          <button
            class="dataset-card ${{selected ? "selected" : ""}}"
            type="button"
            data-drilldown-focus="${{esc(row.effective_id)}}"
            style="--dataset-color:${{color}}"
          >
            <div class="dataset-card-head">
              <strong>${{esc(row.dataset_key)}}</strong>
              <span class="dataset-card-metric">${{esc(row.primary_metric_name || "primary")}} ${{fmt(row.primary_metric_value)}}</span>
            </div>
            <div class="dataset-card-meta">
              ${{esc(row.selection_metric_name || "selection")}} ${{fmt(row.selection_metric_value)}} · best epoch ${{esc(String(row.best_epoch ?? "-"))}} · suite ${{esc(row.suite_name)}}
            </div>
          </button>
        `;
      }}).join("");
    }}

    function renderExplorerCurves(rows, focusRow) {{
      const mode = STATE.explorerCurveMode;
      const usable = rows.filter((row) => Array.isArray(row.curve_series) && row.curve_series.length);
      if (!usable.length) {{
        renderEmptyChart("explorer-curves", "Curves are not available for the selected leaderboard entry.");
        renderLegend("explorer-curves-legend", []);
        return;
      }}
      const width = 1140;
      const height = 396;
      const margin = {{ top: 48, right: 22, bottom: 56, left: 62 }};
      const plotWidth = width - margin.left - margin.right;
      const plotHeight = height - margin.top - margin.bottom;
      const maxEpoch = Math.max(...usable.flatMap((row) => row.curve_series.map((point) => point.epoch || 0)), 1);
      const values = usable.flatMap((row) => row.curve_series.map((point) => num(point[mode])).filter((value) => value !== null));
      if (!values.length) {{
        renderEmptyChart("explorer-curves", "The selected curve mode is not available for these runs.");
        renderLegend("explorer-curves-legend", []);
        return;
      }}
      const bounds = curveAxisBounds(values, mode);
      const x = (epoch) => maxEpoch <= 1 ? margin.left + plotWidth / 2 : margin.left + ((epoch - 1) / (maxEpoch - 1)) * plotWidth;
      const y = (value) => margin.top + (1 - ((value - bounds.minValue) / ((bounds.maxValue - bounds.minValue) || 1))) * plotHeight;
      const title = `${{STATE.leaderboardModel || "Model"}} · ${{splitLabel(STATE.leaderboardSplitMode)}} · ${{mode.replace("_", " ")}}`;
      const svg = [
        svgFrame(width, height, "Training curves"),
        `<text x="${{margin.left}}" y="24" fill="#172026" font-size="15" font-weight="700">${{esc(title)}}</text>`,
        `<text x="${{margin.left}}" y="42" fill="#5c666d" font-size="12">Focused observation window: one best filtered run per dataset for the pinned model and split.</text>`,
      ];
      for (let tickIndex = 0; tickIndex <= 4; tickIndex += 1) {{
        const raw = bounds.minValue + ((bounds.maxValue - bounds.minValue) * tickIndex) / 4;
        const py = y(raw);
        svg.push(`<line x1="${{margin.left}}" y1="${{py}}" x2="${{width - margin.right}}" y2="${{py}}" stroke="#e5e2d9"></line>`);
        svg.push(`<text x="${{margin.left - 8}}" y="${{py + 4}}" text-anchor="end" fill="#5c666d" font-size="11">${{fmt(raw, mode === "lr" ? 5 : 3)}}</text>`);
      }}
      const epochTicks = [...new Set([1, Math.max(1, Math.round(maxEpoch / 2)), maxEpoch])].sort((a, b) => a - b);
      epochTicks.forEach((epoch) => {{
        const px = x(epoch);
        svg.push(`<line x1="${{px}}" y1="${{margin.top}}" x2="${{px}}" y2="${{height - margin.bottom + 4}}" stroke="#ece7dc"></line>`);
        svg.push(`<text x="${{px}}" y="${{height - 16}}" text-anchor="middle" fill="#5c666d" font-size="11">ep ${{epoch}}</text>`);
      }});
      const legendItems = [];
      usable.forEach((row) => {{
        const selected = focusRow && row.effective_id === focusRow.effective_id;
        const color = datasetColor(row.dataset_key);
        const strokeWidth = selected ? 4.2 : 1.9;
        const opacity = selected ? 1 : 0.18;
        legendItems.push({{ label: `${{row.dataset_key}} · ${{fmt(row.primary_metric_value)}}`, color }});
        const points = row.curve_series
          .map((point) => {{
            const value = num(point[mode]);
            return value === null ? "" : `${{x(point.epoch)}},${{y(value)}}`;
          }})
          .filter(Boolean);
        if (!points.length) return;
        svg.push(`<g data-drilldown-focus="${{esc(row.effective_id)}}" style="cursor:pointer">`);
        svg.push(`<polyline fill="none" stroke="transparent" stroke-width="12" stroke-linecap="round" stroke-linejoin="round" points="${{points.join(" ")}}"></polyline>`);
        if (points.length >= 2) {{
          if (selected) {{
            svg.push(`<polyline fill="none" stroke="#ffffff" stroke-width="7.5" stroke-linecap="round" stroke-linejoin="round" opacity="0.94" points="${{points.join(" ")}}"></polyline>`);
          }}
          svg.push(`<polyline fill="none" stroke="${{color}}" stroke-width="${{strokeWidth}}" stroke-linecap="round" stroke-linejoin="round" opacity="${{opacity}}" points="${{points.join(" ")}}"></polyline>`);
        }}
        row.curve_series.forEach((point) => {{
          const value = num(point[mode]);
          if (value === null) return;
          const radius = selected ? (point.is_best ? 5.0 : 3.3) : (point.is_best ? 4.2 : 2.5);
          svg.push(`<circle cx="${{x(point.epoch)}}" cy="${{y(value)}}" r="${{radius}}" fill="${{color}}" opacity="${{selected ? 1 : 0.26}}"></circle>`);
        }});
        svg.push("</g>");
      }});
      svg.push("</svg>");
      document.getElementById("explorer-curves").innerHTML = svg.join("");
      renderLegend("explorer-curves-legend", legendItems);
    }}

    function renderExplorerHistory(focusRow) {{
      const target = document.getElementById("explorer-history");
      if (!focusRow || !Array.isArray(focusRow.source_history) || !focusRow.source_history.length) {{
        target.innerHTML = `<div class="subtle">No source history available.</div>`;
        return;
      }}
      target.innerHTML = focusRow.source_history.map((item) => `
        <article class="history-item ${{item.is_selected ? "selected" : ""}}">
          <strong>${{esc(item.suite_name)}}</strong>
          <div class="subtle">${{esc(STATUS_LABELS[item.status_bucket] || item.status_bucket)}} · run ${{esc(item.run_id || "-")}}</div>
          <div class="subtle">Primary ${{fmt(item.primary_metric_value)}} · selection ${{fmt(item.selection_metric_value)}} · ${{esc(fmtDate(item.finished_at))}}</div>
        </article>
      `).join("");
    }}

    function renderExplorerTable(rows, focusRow) {{
      const body = document.getElementById("explorer-table-body");
      if (!rows.length) {{
        body.innerHTML = `<tr><td colspan="7" class="subtle">No rows are available for this explorer selection.</td></tr>`;
        return;
      }}
      body.innerHTML = rows.map((row) => `
        <tr style="${{focusRow && row.effective_id === focusRow.effective_id ? "background:#f6f2e7;" : ""}}">
          <td>${{esc(row.dataset_key)}}</td>
          <td>${{esc(row.suite_name)}}</td>
          <td>${{esc(STATUS_LABELS[row.status_bucket] || row.status_bucket)}}</td>
          <td>${{fmt(row.primary_metric_value)}}</td>
          <td>${{fmt(row.selection_metric_value)}}</td>
          <td>${{esc(String(row.best_epoch ?? "-"))}}</td>
          <td class="mono">${{esc(row.condition_id)}}</td>
        </tr>
      `).join("");
    }}

    function renderExplorer(rows) {{
      const selectedRows = drilldownRows(rows);
      const focusRow = syncDrilldownFocus(selectedRows);
      const entry = selectedLeaderboardEntry(rows);

      renderExplorerCurves(selectedRows, focusRow);
      renderDrilldownDatasetCards(selectedRows, focusRow);
      renderExplorerHistory(focusRow);
      renderExplorerTable(selectedRows, focusRow);
      bindDrilldownFocusControls();

      document.getElementById("drilldown-selection-title").textContent = entry ? `${{entry.model}} · ${{splitLabel(STATE.leaderboardSplitMode)}}` : "-";
      document.getElementById("drilldown-selection-meta").textContent = entry
        ? "Best filtered effective run per dataset for the pinned leaderboard row. Click a curve or dataset card to focus one dataset."
        : "Choose a leaderboard bar to load dataset-level curves.";
      document.getElementById("drilldown-selection-pills").innerHTML = entry ? [
        `<span class="selection-pill">avg rank ${{fmt(entry.avg_rank, 2)}}</span>`,
        `<span class="selection-pill">wins ${{entry.win_count}} · podium ${{entry.podium_count}}</span>`,
        `<span class="selection-pill">${{esc(entry.selection_metric_name || currentSelectionMetricLabel(selectedRows))}} ${{fmt(entry.avg_selection_metric_value)}}</span>`,
        `<span class="selection-pill">${{entry.datasets_used}} dataset(s)</span>`,
        focusRow ? `<span class="selection-pill">focus ${{esc(focusRow.dataset_key)}}</span>` : "",
      ].filter(Boolean).join("") : "";

      document.getElementById("explorer-primary-metric").textContent = entry ? `${{entry.selection_metric_name || currentSelectionMetricLabel(selectedRows)}} ${{fmt(entry.avg_selection_metric_value)}}` : "-";
      document.getElementById("explorer-model-count").textContent = String(selectedRows.length);
      document.getElementById("explorer-status").textContent = focusRow ? shortLabel(focusRow.dataset_key.replace(":", " / "), 24) : "-";
      document.getElementById("explorer-source-count").textContent = focusRow ? String(focusRow.source_count || 0) : "0";
      renderComparisonSection(rows, focusRow);
    }}

    function comparisonSeriesKey(row) {{
      return STATE.comparisonView === "dataset_across_models" ? row.model : row.dataset_key;
    }}

    function comparisonSeriesColor(row) {{
      return STATE.comparisonView === "dataset_across_models" ? modelColor(row.model) : datasetColor(row.dataset_key);
    }}

    function comparisonAnchorOptions(rows) {{
      if (STATE.comparisonView === "dataset_across_models") {{
        return sortByOrder(
          [...new Set(rows.map((row) => row.dataset_key).filter(Boolean))],
          APP.dataset_order
        );
      }}
      return sortByOrder(
        [...new Set(rows.map((row) => row.model).filter(Boolean))],
        APP.model_order
      );
    }}

    function syncComparisonAnchor(rows, focusRow) {{
      const options = comparisonAnchorOptions(rows);
      let desired = STATE.comparisonAnchor;
      if (!options.includes(desired)) {{
        desired = STATE.comparisonView === "dataset_across_models"
          ? ((focusRow && options.includes(focusRow.dataset_key)) ? focusRow.dataset_key : (options[0] || ""))
          : ((STATE.leaderboardModel && options.includes(STATE.leaderboardModel)) ? STATE.leaderboardModel : (options[0] || ""));
      }}
      STATE.comparisonAnchor = desired;
      return options;
    }}

    function comparisonRows(rows) {{
      const rankable = rows.filter((row) => row.is_rankable);
      if (!STATE.comparisonAnchor) return [];
      const topK = Math.max(1, num(STATE.comparisonTopK) ?? 5);
      if (STATE.comparisonView === "dataset_across_models") {{
        return bestByModel(rankable.filter((row) => row.dataset_key === STATE.comparisonAnchor)).slice(0, topK);
      }}
      return bestByDataset(rankable.filter((row) => row.model === STATE.comparisonAnchor))
        .sort(compareRows)
        .slice(0, topK);
    }}

    function syncComparisonFocus(rows, focusRow) {{
      const available = rows.map((row) => comparisonSeriesKey(row));
      let desired = STATE.comparisonFocusKey;
      if (!available.includes(desired)) {{
        desired = STATE.comparisonView === "dataset_across_models"
          ? ((STATE.leaderboardModel && available.includes(STATE.leaderboardModel)) ? STATE.leaderboardModel : (available[0] || ""))
          : ((focusRow && available.includes(focusRow.dataset_key)) ? focusRow.dataset_key : (available[0] || ""));
      }}
      STATE.comparisonFocusKey = desired;
      return desired;
    }}

    function renderComparisonControls(rows, focusRow) {{
      const anchorLabel = STATE.comparisonView === "dataset_across_models" ? "Dataset" : "Model";
      const options = syncComparisonAnchor(rows, focusRow);
      document.getElementById("comparison-view").value = STATE.comparisonView;
      document.getElementById("comparison-topk").value = String(STATE.comparisonTopK);
      document.getElementById("comparison-curve-mode").value = STATE.comparisonCurveMode;
      document.getElementById("comparison-anchor-label").textContent = anchorLabel;
      const select = document.getElementById("comparison-anchor");
      select.innerHTML = options.map((value) => `<option value="${{esc(value)}}">${{esc(value)}}</option>`).join("");
      select.value = STATE.comparisonAnchor;
    }}

    function bindComparisonFocusControls() {{
      document.querySelectorAll("[data-comparison-focus]").forEach((node) => {{
        node.addEventListener("click", () => {{
          STATE.comparisonFocusKey = node.dataset.comparisonFocus;
          const rankableRows = filteredRankableRows();
          const focusRow = syncDrilldownFocus(drilldownRows(rankableRows));
          renderComparisonSection(rankableRows, focusRow);
        }});
      }});
    }}

    function renderComparisonSection(rows, focusRow) {{
      renderComparisonControls(rows, focusRow);
      const selectedRows = comparisonRows(rows);
      const usable = selectedRows.filter((row) => Array.isArray(row.curve_series) && row.curve_series.length);
      if (!usable.length) {{
        renderEmptyChart("comparison-curves", "No comparable curve set is available under the current filters.");
        renderLegend("comparison-curves-legend", []);
        document.getElementById("comparison-note").textContent = "Widen the filters or switch the comparison perspective.";
        return;
      }}

      const activeKey = syncComparisonFocus(usable, focusRow);
      const mode = STATE.comparisonCurveMode;
      const width = 1180;
      const height = 404;
      const margin = {{ top: 48, right: 22, bottom: 56, left: 62 }};
      const plotWidth = width - margin.left - margin.right;
      const plotHeight = height - margin.top - margin.bottom;
      const maxEpoch = Math.max(...usable.flatMap((row) => row.curve_series.map((point) => point.epoch || 0)), 1);
      const values = usable.flatMap((row) => row.curve_series.map((point) => num(point[mode])).filter((value) => value !== null));
      if (!values.length) {{
        renderEmptyChart("comparison-curves", "The selected curve mode is not available for this comparison slice.");
        renderLegend("comparison-curves-legend", []);
        document.getElementById("comparison-note").textContent = "";
        return;
      }}
      const bounds = curveAxisBounds(values, mode);
      const x = (epoch) => maxEpoch <= 1 ? margin.left + plotWidth / 2 : margin.left + ((epoch - 1) / (maxEpoch - 1)) * plotWidth;
      const y = (value) => margin.top + (1 - ((value - bounds.minValue) / ((bounds.maxValue - bounds.minValue) || 1))) * plotHeight;
      const title = STATE.comparisonView === "dataset_across_models"
        ? `${{STATE.comparisonAnchor || "Dataset"}} · top-${{usable.length}} models`
        : `${{STATE.comparisonAnchor || "Model"}} · top-${{usable.length}} datasets`;
      const subtitle = STATE.comparisonView === "dataset_across_models"
        ? "Same dataset, different models. Other settings continue to follow the current filters."
        : "Same model, different datasets. Other settings continue to follow the current filters.";
      const svg = [
        svgFrame(width, height, "Comparison curves"),
        `<text x="${{margin.left}}" y="24" fill="#172026" font-size="15" font-weight="700">${{esc(title)}}</text>`,
        `<text x="${{margin.left}}" y="42" fill="#5c666d" font-size="12">${{esc(subtitle)}}</text>`,
      ];
      for (let tickIndex = 0; tickIndex <= 4; tickIndex += 1) {{
        const raw = bounds.minValue + ((bounds.maxValue - bounds.minValue) * tickIndex) / 4;
        const py = y(raw);
        svg.push(`<line x1="${{margin.left}}" y1="${{py}}" x2="${{width - margin.right}}" y2="${{py}}" stroke="#e5e2d9"></line>`);
        svg.push(`<text x="${{margin.left - 8}}" y="${{py + 4}}" text-anchor="end" fill="#5c666d" font-size="11">${{fmt(raw, mode === "lr" ? 5 : 3)}}</text>`);
      }}
      const epochTicks = [...new Set([1, Math.max(1, Math.round(maxEpoch / 2)), maxEpoch])].sort((a, b) => a - b);
      epochTicks.forEach((epoch) => {{
        const px = x(epoch);
        svg.push(`<line x1="${{px}}" y1="${{margin.top}}" x2="${{px}}" y2="${{height - margin.bottom + 4}}" stroke="#ece7dc"></line>`);
        svg.push(`<text x="${{px}}" y="${{height - 16}}" text-anchor="middle" fill="#5c666d" font-size="11">ep ${{epoch}}</text>`);
      }});
      const legendItems = [];
      usable.forEach((row) => {{
        const key = comparisonSeriesKey(row);
        const selected = key === activeKey;
        const color = comparisonSeriesColor(row);
        const points = row.curve_series
          .map((point) => {{
            const value = num(point[mode]);
            return value === null ? "" : `${{x(point.epoch)}},${{y(value)}}`;
          }})
          .filter(Boolean);
        if (!points.length) return;
        legendItems.push({{
          label: `${{key}} · ${{fmt(row.primary_metric_value)}}`,
          color,
        }});
        svg.push(`<g data-comparison-focus="${{esc(key)}}" style="cursor:pointer">`);
        svg.push(`<polyline fill="none" stroke="transparent" stroke-width="12" stroke-linecap="round" stroke-linejoin="round" points="${{points.join(" ")}}"></polyline>`);
        if (points.length >= 2) {{
          if (selected) {{
            svg.push(`<polyline fill="none" stroke="#ffffff" stroke-width="7.5" stroke-linecap="round" stroke-linejoin="round" opacity="0.94" points="${{points.join(" ")}}"></polyline>`);
          }}
          svg.push(`<polyline fill="none" stroke="${{color}}" stroke-width="${{selected ? 4.2 : 1.9}}" stroke-linecap="round" stroke-linejoin="round" opacity="${{selected ? 1 : 0.14}}" points="${{points.join(" ")}}"></polyline>`);
        }}
        row.curve_series.forEach((point) => {{
          const value = num(point[mode]);
          if (value === null) return;
          svg.push(`<circle cx="${{x(point.epoch)}}" cy="${{y(value)}}" r="${{selected ? (point.is_best ? 5.0 : 3.2) : (point.is_best ? 4.0 : 2.3)}}" fill="${{color}}" opacity="${{selected ? 1 : 0.22}}"></circle>`);
        }});
        svg.push("</g>");
      }});
      svg.push("</svg>");
      document.getElementById("comparison-curves").innerHTML = svg.join("");
      renderLegend("comparison-curves-legend", legendItems);
      document.getElementById("comparison-note").textContent = `Showing top ${{usable.length}} series under the current filters. Click a curve to pin it and fade the rest for easier comparison.`;
      bindComparisonFocusControls();
    }}

    function refresh() {{
      syncFilterControls();
      updateMatchingCounts();
      const rankableRows = filteredRankableRows();
      const columns = syncLeaderboardSelection(rankableRows);
      renderOverallRanking(rankableRows, columns);
      renderSplitCurves(rankableRows, columns);
      renderHeatmap(rankableRows);
    }}

    let resizeDebounceHandle = 0;

    function bindResponsiveRerender() {{
      window.addEventListener("resize", () => {{
        window.clearTimeout(resizeDebounceHandle);
        resizeDebounceHandle = window.setTimeout(() => {{
          renderOverviewCharts();
          refresh();
        }}, 120);
      }});
    }}

    function bindControls() {{
      document.getElementById("reset-filters").addEventListener("click", () => {{
        FILTER_FIELDS.forEach((field) => {{
          STATE.filters[field.key] = new Set();
        }});
        STATE.hoverSplitMode = "";
        STATE.hoverModel = "";
        refresh();
      }});
      document.getElementById("split-curve-mode").addEventListener("change", (event) => {{
        STATE.explorerCurveMode = event.target.value;
        refresh();
      }});
    }}

    function init() {{
      FILTER_FIELDS.forEach((field) => {{
        STATE.filters[field.key] = new Set();
      }});
      renderFilters();
      bindControls();
      bindResponsiveRerender();
      renderOverviewCharts();
      STATE.explorerCurveMode = document.getElementById("split-curve-mode").value;
      refresh();
    }}

    init();
  </script>
</body>
</html>
"""


class ResultsSiteBuilder:
    def __init__(self, suite_dirs: list[Path], site_name: str, output_dir: Path) -> None:
        self.suite_dirs = suite_dirs
        self.site_name = site_name
        self.output_dir = output_dir

    def build(self) -> tuple[Path, dict[str, Any]]:
        ensure_dir(self.output_dir)
        suites = load_suite_sources(self.suite_dirs)
        discovered_specs = discover_argument_specs()
        raw_records = build_run_records(suites)
        dataset_primary_metrics = derive_primary_metrics(raw_records)
        attach_primary_metrics(raw_records, dataset_primary_metrics)
        effective_records, selected_raw_map = build_effective_records(raw_records)

        raw_rows: list[dict[str, Any]] = []
        for record in raw_records:
            row = record.to_payload_dict()
            row["is_selected_effective"] = record.raw_id in selected_raw_map
            row["effective_id"] = selected_raw_map.get(record.raw_id, "")
            raw_rows.append(row)

        effective_rows = [record.to_payload_dict() for record in effective_records]
        filter_schema = [asdict(field) for field in build_filter_schema(discovered_specs, raw_records, suites)]
        model_order = []
        dataset_order = []
        for row in effective_rows:
            if row.get("model") and row["model"] not in model_order:
                model_order.append(row["model"])
            if row.get("dataset_key") and row["dataset_key"] not in dataset_order:
                dataset_order.append(row["dataset_key"])

        summary = build_summary(self.site_name, suites, raw_rows, effective_rows)
        payload = SitePayload(
            site_name=self.site_name,
            generated_at=datetime.now().isoformat(timespec="seconds"),
            summary=summary,
            suites=[suite.metadata() for suite in suites],
            filters=filter_schema,
            dataset_primary_metrics=dataset_primary_metrics,
            model_order=model_order,
            dataset_order=dataset_order,
            raw_runs=raw_rows,
            effective_runs=effective_rows,
        )

        dashboard_path = self.output_dir / "dashboard.html"
        dashboard_json_path = self.output_dir / "dashboard_data.json"
        raw_csv_path = self.output_dir / "raw_runs.csv"
        effective_csv_path = self.output_dir / "effective_runs.csv"
        metadata_path = self.output_dir / "metadata.json"

        dashboard_path.write_text(render_dashboard(payload), encoding="utf-8")
        dashboard_json_path.write_text(json.dumps(asdict(payload), ensure_ascii=False, indent=2), encoding="utf-8")
        write_csv(raw_csv_path, raw_rows)
        write_csv(effective_csv_path, effective_rows)
        metadata = {
            "generated_at": payload.generated_at,
            "site_name": self.site_name,
            "output_dir": str(self.output_dir),
            "suite_dirs": [str(path) for path in self.suite_dirs],
            "summary": summary,
            "dataset_primary_metrics": dataset_primary_metrics,
            "filter_keys": [field["key"] for field in filter_schema],
            "suite_metadata": [suite.metadata() for suite in suites],
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return self.output_dir, summary


def default_site_name(suite_dirs: list[Path]) -> str:
    if not suite_dirs:
        return "results_dashboard"
    if len(suite_dirs) == 1:
        return slugify(suite_dirs[0].name)
    joined = "__".join(slugify(path.name) for path in suite_dirs[:3])
    if len(suite_dirs) > 3:
        joined = f"{joined}__plus_{len(suite_dirs) - 3}"
    return joined


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a static HTML results dashboard from one or more batch suite outputs.")
    parser.add_argument(
        "--suite-dir",
        action="append",
        required=True,
        help="Path to a batch output directory such as experiments/output/smoke_suite. Repeat for multiple suites.",
    )
    parser.add_argument(
        "--site-name",
        type=str,
        default=None,
        help="Site name and default output leaf directory. Defaults to the suite name for a single input.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Exact output directory. Defaults to results/output/<site-name>/.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    suite_dirs = [Path(path).expanduser().resolve() for path in args.suite_dir]
    for suite_dir in suite_dirs:
        if not suite_dir.exists():
            raise FileNotFoundError(f"Suite directory does not exist: {suite_dir}")
        if not suite_dir.is_dir():
            raise ValueError(f"Suite directory is not a directory: {suite_dir}")

    site_name = slugify(args.site_name) if args.site_name else default_site_name(suite_dirs)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (DEFAULT_OUTPUT_ROOT / site_name)

    builder = ResultsSiteBuilder(suite_dirs=suite_dirs, site_name=site_name, output_dir=output_dir)
    built_dir, summary = builder.build()
    print(f"Wrote dashboard to {built_dir / 'dashboard.html'}")
    print(f"Raw runs: {summary['raw_run_count']} | Effective runs: {summary['effective_run_count']} | Rankable: {summary['rankable_effective_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
