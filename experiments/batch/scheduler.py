#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from pathlib import Path

from data.channels import PUBLIC_CHANNEL_SYSTEM_CHOICES
from data.finetune.core import get_subject_ids, resolve_loso_target
from experiments.batch.suite_config import (
    DEFAULT_DATASET_ORDER,
    DEFAULT_MODEL_ORDER,
    iter_sweep_arg_maps,
    resolve_suite_spec,
    resolve_task_run_args,
    suite_spec_to_dict,
)
from experiments.common.result_schema import TASK_STATUS_HEADERS
from experiments.common.runtime_metadata import (
    extract_final_metric_fields,
    resolve_data_config_fields,
    split_dataset_task,
    stable_json_text,
    subject_list_text,
)


BASE_RUN_ARG_ORDER = [
    "epochs",
    "batch_size",
    "num_workers",
    "data_ratio",
    "seed",
    "split_mode",
    "held_out_subject",
    "fold",
    "channel_system",
    "channel_variant",
    "channel_count_target",
    "channel_sparsity_config",
    "classifier_lr_scale",
    "early_stopping_patience",
    "save_checkpoints",
]

BOOLEAN_VALUE_ARG_KEYS = frozenset({
    "use_pretrained_weights",
    "frozen",
})

RESUME_RELOCATABLE_RUN_ARG_KEYS = frozenset({
    "cache_dir",
})

RESUME_REPO_RELATIVE_RUN_ARG_KEYS = frozenset({
    "pipeline",
})

_EXPERIMENT_ID_PATTERN = re.compile(r"^\s*Experiment ID:\s*(.+?)\s*$")
_RUN_ID_PATTERN = re.compile(r"^\s*Run ID:\s*(.+?)\s*$")
_CONFIG_ID_PATTERN = re.compile(r"^\s*Config ID:\s*(.+?)\s*$")


def now_text():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def json_dump(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp_path.replace(path)


def json_load(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_cache_link_fields(cache_usage):
    cache_usage = cache_usage or {}
    locator_usage = cache_usage.get("locator") or {}
    hdf5_usage = cache_usage.get("hdf5") or {}
    return {
        "cache_root": clean_optional_text(cache_usage.get("cache_root")),
        "locator_cache_id": clean_optional_text(locator_usage.get("cache_id")),
        "hdf5_cache_id": clean_optional_text(hdf5_usage.get("cache_id")),
    }


def format_job_value(value):
    if value is None:
        return "default"

    if isinstance(value, float) and value.is_integer():
        text = str(int(value))
    else:
        text = str(value)

    safe_chars = []
    for char in text:
        if char.isalnum() or char in {"-", "_"}:
            safe_chars.append(char)
        elif char == ".":
            safe_chars.append("p")
        else:
            safe_chars.append("_")
    return "".join(safe_chars)


def is_default_data_ratio(value):
    if value is None:
        return True
    try:
        return float(value) == 1.0
    except (TypeError, ValueError):
        return False


def parse_int_or_none(text):
    try:
        return int(str(text).strip())
    except (TypeError, ValueError):
        return None


def parse_float_or_none(text):
    try:
        return float(str(text).strip())
    except (TypeError, ValueError):
        return None


def format_duration_brief(seconds):
    if seconds is None:
        return "0s"
    seconds = max(0, int(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m"
    if minutes > 0:
        return f"{minutes}m{sec:02d}s"
    return f"{sec}s"


def clean_optional_text(value):
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def truncate_text(text, width):
    text = str(text)
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."


def discover_repo_root():
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (
            (
                (parent / "run_single_experiment.py").exists()
                or (parent / "run_finetune.py").exists()
            )
            and (
                (parent / "models" / "registry.py").exists()
                or (parent / "model_registry").is_dir()
            )
        ):
            return parent
    raise RuntimeError("Could not locate repository root from experiments/batch/scheduler.py")


@dataclass
class Task:
    index: int
    model: str
    dataset: str
    run_args: dict
    explicit_name: str | None = None

    @property
    def job_name(self):
        if self.explicit_name:
            return f"{self.index:04d}__{format_job_value(self.explicit_name)}"

        split_mode = format_job_value(self.run_args["split_mode"])
        fold = self.run_args["fold"]
        held_out_subject = clean_optional_text(self.run_args.get("held_out_subject"))
        seed = self.run_args["seed"]
        channel_system = format_job_value(self.run_args.get("channel_system"))
        channel_variant = clean_optional_text(self.run_args.get("channel_variant"))
        classifier_lr_scale = format_job_value(self.run_args.get("classifier_lr_scale"))
        data_ratio = self.run_args.get("data_ratio")
        data_ratio_suffix = ""
        if not is_default_data_ratio(data_ratio):
            data_ratio_suffix = f"__dr{format_job_value(data_ratio)}"
        channel_variant_suffix = ""
        if channel_variant is not None:
            channel_variant_suffix = f"__cv{format_job_value(channel_variant)}"
        if split_mode == "loso" and held_out_subject is not None:
            split_suffix = f"__{split_mode}_s{format_job_value(held_out_subject)}"
        else:
            split_suffix = f"__{split_mode}_f{fold}"
        return (
            f"{self.index:04d}__{self.model}__{self.dataset}"
            f"{split_suffix}"
            f"__ch{channel_system}"
            f"{channel_variant_suffix}"
            f"__clr{classifier_lr_scale}"
            f"{data_ratio_suffix}"
            f"__seed{seed}"
        )

    def manifest_payload(self):
        return {
            "index": self.index,
            "job_name": self.job_name,
            "explicit_name": self.explicit_name,
            "model": self.model,
            "dataset": self.dataset,
            "run_args": dict(self.run_args),
        }


class Scheduler:
    def __init__(self, args):
        self.args = args
        if not hasattr(self.args, "processes_per_gpu"):
            self.args.processes_per_gpu = 1
        if self.args.processes_per_gpu < 1:
            raise ValueError("processes_per_gpu must be >= 1.")
        self.repo_root = discover_repo_root()
        self.suite_spec = getattr(args, "_resolved_suite_spec", None)
        if self.suite_spec is None:
            self.suite_spec = resolve_suite_spec(args, parser=None, argv=getattr(args, "_argv", []))
        self.output_dir = Path(args.output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.scheduler_dir = self.output_dir / "scheduler"
        self.scheduler_dir.mkdir(parents=True, exist_ok=True)

        if args.scheduler_log:
            self.scheduler_log_path = Path(args.scheduler_log).resolve()
        else:
            self.scheduler_log_path = self.scheduler_dir / "scheduler.log"
        self.scheduler_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.scheduler_log_file = self.scheduler_log_path.open("a", encoding="utf-8")
        self._dashboard_enabled = sys.stdout.isatty()
        self._dashboard_cursor_hidden = False

        self.launcher_log_dir = self.scheduler_dir / "launcher_logs"
        self.launcher_log_dir.mkdir(parents=True, exist_ok=True)

        self.state_path = self.scheduler_dir / "state.json"
        self.summary_path = self.scheduler_dir / "summary.json"
        self.task_manifest_path = self.scheduler_dir / "task_manifest.json"
        self.task_status_csv_path = self.scheduler_dir / "task_status.csv"
        self._loso_subject_ids_cache = {}
        self._split_probe_order = []
        self._split_probe_descriptors = {}
        self._prepare_split_probe_plan()

        try:
            self.all_tasks = self.build_tasks()
        except Exception:
            self.scheduler_log_file.flush()
            self.scheduler_log_file.close()
            raise
        self.pending_tasks = deque()
        self.total_tasks = len(self.all_tasks)
        self.completed_tasks = []
        self.failed_tasks = []
        self.carried_completed_tasks = []
        self.carried_failed_tasks = []
        self.prior_manifest_repo_root = None
        self.prior_manifest_tasks = {}
        self.prior_task_rows = {}
        self.prior_attempt_counts = {}
        self.resume_actions_by_job = {
            task.job_name: "run"
            for task in self.all_tasks
        }
        resume_ignore_run_args = {
            clean_optional_text(key)
            for key in (getattr(args, "resume_ignore_run_args", None) or [])
            if clean_optional_text(key) is not None
        }
        if bool(getattr(args, "resume_relaxed_host_paths", False)):
            resume_ignore_run_args.update(RESUME_RELOCATABLE_RUN_ARG_KEYS)
        self.resume_ignore_run_args = frozenset(resume_ignore_run_args)
        self.resume_metadata = {
            "enabled": bool(getattr(args, "resume", False)),
            "skip_completed": bool(getattr(args, "skip_completed", True)),
            "retry_failed": bool(getattr(args, "retry_failed", False)),
            "relaxed_host_paths": bool(getattr(args, "resume_relaxed_host_paths", False)),
            "ignored_run_args": sorted(self.resume_ignore_run_args),
        }
        self.gpu_slots = self._build_gpu_slots()
        self.active = {}
        self.stop_requested = False
        self._gpu_telemetry_error = None
        self._run_status_cache = {}
        self._run_final_metrics_cache = {}

        if self.resume_metadata["enabled"]:
            self._apply_resume_plan()
        else:
            self.pending_tasks = deque(self._order_tasks_for_launch(self.all_tasks))

        signal.signal(signal.SIGINT, self.handle_stop)
        signal.signal(signal.SIGTERM, self.handle_stop)

        self.write_manifest()
        self.write_state()
        self._setup_dashboard_console()

    def _all_completed_records(self):
        return [*self.carried_completed_tasks, *self.completed_tasks]

    def _all_failed_records(self):
        return [*self.carried_failed_tasks, *self.failed_tasks]

    def _build_gpu_slots(self):
        return [
            (gpu_id, slot_index)
            for gpu_id in self.args.gpus
            for slot_index in range(self.args.processes_per_gpu)
        ]

    def _format_gpu_slot(self, gpu_id, slot_index):
        if self.args.processes_per_gpu <= 1:
            return str(gpu_id)
        return f"{gpu_id}:{slot_index + 1}"

    def _normalize_manifest_task_payload(self, payload):
        return {
            "index": payload.get("index"),
            "job_name": payload.get("job_name"),
            "explicit_name": payload.get("explicit_name"),
            "model": payload.get("model"),
            "dataset": payload.get("dataset"),
            "run_args": dict(payload.get("run_args") or {}),
        }

    def _current_manifest_tasks(self):
        return [task.manifest_payload() for task in self.all_tasks]

    @staticmethod
    def _normalize_resume_repo_relative_path(text, source_repo_root):
        text = clean_optional_text(text)
        if text is None:
            return None

        path = Path(text)
        if source_repo_root is not None and path.is_absolute():
            try:
                return f"repo://{path.relative_to(source_repo_root).as_posix()}"
            except ValueError:
                pass

        return f"repo://{path.as_posix()}"

    def _normalize_resume_run_arg_value(self, key, value, source_repo_root):
        if isinstance(value, dict):
            return {
                item_key: self._normalize_resume_run_arg_value(
                    item_key,
                    item_value,
                    source_repo_root,
                )
                for item_key, item_value in value.items()
            }
        if isinstance(value, list):
            return [
                self._normalize_resume_run_arg_value(key, item, source_repo_root)
                for item in value
            ]

        if (
            self.resume_metadata["relaxed_host_paths"]
            and key in RESUME_REPO_RELATIVE_RUN_ARG_KEYS
        ):
            return self._normalize_resume_repo_relative_path(value, source_repo_root)

        return value

    def _normalize_resume_run_args(self, run_args, source_repo_root):
        normalized = {}
        for key, value in dict(run_args or {}).items():
            key_text = clean_optional_text(key) or key
            if key_text in self.resume_ignore_run_args:
                continue
            normalized[key] = self._normalize_resume_run_arg_value(
                key_text,
                value,
                source_repo_root,
            )
        return normalized

    def _task_identity_from_payload(self, payload):
        explicit_name = clean_optional_text(payload.get("explicit_name"))
        if explicit_name is not None:
            return f"explicit::{explicit_name}"

        job_name = clean_optional_text(payload.get("job_name"))
        if job_name is None:
            raise ValueError("Task payload is missing both explicit_name and job_name.")
        return f"job::{job_name}"

    def _task_definition_from_payload(self, payload, source_repo_root=None):
        return {
            "explicit_name": clean_optional_text(payload.get("explicit_name")),
            "model": payload.get("model"),
            "dataset": payload.get("dataset"),
            "run_args": self._normalize_resume_run_args(
                payload.get("run_args"),
                source_repo_root,
            ),
        }

    def _index_manifest_tasks(self, payloads, source_name):
        indexed = {}
        for payload in payloads:
            normalized = self._normalize_manifest_task_payload(payload)
            identity = self._task_identity_from_payload(normalized)
            if identity in indexed:
                raise ValueError(
                    f"Duplicate resume task identity '{identity}' in {source_name}. "
                    "For explicit task mode, make task names unique when using --resume."
                )
            indexed[identity] = normalized
        return indexed

    def _load_resume_manifest(self):
        if not self.task_manifest_path.exists():
            raise FileNotFoundError(
                f"--resume requires an existing task manifest under the same output directory: {self.task_manifest_path}"
            )

        payload = json_load(self.task_manifest_path)
        prior_repo_root = clean_optional_text(payload.get("repo_root"))
        self.prior_manifest_repo_root = Path(prior_repo_root) if prior_repo_root is not None else None
        prior_tasks = payload.get("tasks")
        if not isinstance(prior_tasks, list):
            raise ValueError(
                f"Invalid resume manifest at {self.task_manifest_path}: missing 'tasks' list."
            )

        return self._index_manifest_tasks(prior_tasks, str(self.task_manifest_path))

    def _load_resume_task_rows(self):
        if not self.task_status_csv_path.exists():
            raise FileNotFoundError(
                f"--resume requires an existing task-status file under the same output directory: {self.task_status_csv_path}"
            )

        rows_by_job = {}
        with self.task_status_csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for raw_row in reader:
                job_name = clean_optional_text(raw_row.get("job_name"))
                if not job_name:
                    continue
                row = {
                    "task_index": parse_int_or_none(raw_row.get("task_index")),
                    "job_name": job_name,
                    "status": clean_optional_text(raw_row.get("status")) or "Pending",
                    "attempt_count": parse_int_or_none(raw_row.get("attempt_count")) or 0,
                    "exp_id": clean_optional_text(raw_row.get("exp_id")),
                    "run_id": clean_optional_text(raw_row.get("run_id")),
                    "config_id": clean_optional_text(raw_row.get("config_id")),
                    "data_config_id": clean_optional_text(raw_row.get("data_config_id")),
                    "data_config_key": clean_optional_text(raw_row.get("data_config_key")),
                    "model": clean_optional_text(raw_row.get("model")),
                    "dataset": clean_optional_text(raw_row.get("dataset")),
                    "task": clean_optional_text(raw_row.get("task")),
                    "split_mode": clean_optional_text(raw_row.get("split_mode")),
                    "held_out_subject": clean_optional_text(raw_row.get("held_out_subject")),
                    "fold": parse_int_or_none(raw_row.get("fold")),
                    "train_subjects": clean_optional_text(raw_row.get("train_subjects")),
                    "val_subjects": clean_optional_text(raw_row.get("val_subjects")),
                    "test_subjects": clean_optional_text(raw_row.get("test_subjects")),
                    "seed": parse_int_or_none(raw_row.get("seed")),
                    "channel_system": clean_optional_text(raw_row.get("channel_system")),
                    "channel_variant": clean_optional_text(raw_row.get("channel_variant")),
                    "channel_sparsity_config": clean_optional_text(raw_row.get("channel_sparsity_config")),
                    "channel_count_target": parse_int_or_none(raw_row.get("channel_count_target")),
                    "channel_count_loaded": parse_int_or_none(raw_row.get("channel_count_loaded")),
                    "input_format": clean_optional_text(raw_row.get("input_format")),
                    "variant": clean_optional_text(raw_row.get("variant")),
                    "epochs": parse_int_or_none(raw_row.get("epochs")),
                    "batch_size": parse_int_or_none(raw_row.get("batch_size")),
                    "data_ratio": parse_float_or_none(raw_row.get("data_ratio")),
                    "train_samples_original": parse_int_or_none(raw_row.get("train_samples_original")),
                    "train_samples": parse_int_or_none(raw_row.get("train_samples")),
                    "val_samples_original": parse_int_or_none(raw_row.get("val_samples_original")),
                    "val_samples": parse_int_or_none(raw_row.get("val_samples")),
                    "test_samples_original": parse_int_or_none(raw_row.get("test_samples_original")),
                    "test_samples": parse_int_or_none(raw_row.get("test_samples")),
                    "classifier_lr_scale": clean_optional_text(raw_row.get("classifier_lr_scale")),
                    "epoch_method": clean_optional_text(raw_row.get("epoch_method")),
                    "pipeline_key": clean_optional_text(raw_row.get("pipeline_key")),
                    "pipeline": clean_optional_text(raw_row.get("pipeline")),
                    "pipeline_params": clean_optional_text(raw_row.get("pipeline_params")),
                    "epoch_params": clean_optional_text(raw_row.get("epoch_params")),
                    "cache_root": clean_optional_text(raw_row.get("cache_root")),
                    "locator_cache_id": clean_optional_text(raw_row.get("locator_cache_id")),
                    "hdf5_cache_id": clean_optional_text(raw_row.get("hdf5_cache_id")),
                    "best_epoch": parse_int_or_none(raw_row.get("best_epoch")),
                    "selection_metric_name": clean_optional_text(raw_row.get("selection_metric_name")),
                    "selection_metric_value": parse_float_or_none(raw_row.get("selection_metric_value")),
                    "final_acc": parse_float_or_none(raw_row.get("final_acc")),
                    "final_f1_macro": parse_float_or_none(raw_row.get("final_f1_macro")),
                    "final_f1_weighted": parse_float_or_none(raw_row.get("final_f1_weighted")),
                    "final_kappa": parse_float_or_none(raw_row.get("final_kappa")),
                    "gpu": parse_int_or_none(raw_row.get("gpu")),
                    "pid": parse_int_or_none(raw_row.get("pid")),
                    "return_code": parse_int_or_none(raw_row.get("return_code")),
                    "started_at": clean_optional_text(raw_row.get("started_at")),
                    "finished_at": clean_optional_text(raw_row.get("finished_at")),
                    "duration_sec": parse_int_or_none(raw_row.get("duration_sec")),
                    "run_relpath": clean_optional_text(raw_row.get("run_relpath")),
                    "config_relpath": clean_optional_text(raw_row.get("config_relpath")),
                    "status_relpath": clean_optional_text(raw_row.get("status_relpath")),
                    "final_test_metrics_relpath": clean_optional_text(raw_row.get("final_test_metrics_relpath")),
                    "log_relpath": clean_optional_text(raw_row.get("log_relpath")),
                    "launcher_log": clean_optional_text(raw_row.get("launcher_log")),
                }
                rows_by_job[job_name] = row
        return rows_by_job

    def _record_from_resume_row(self, task, row):
        return {
            "job_name": task.job_name,
            "explicit_name": task.explicit_name,
            "model": task.model,
            "dataset": task.dataset,
            "run_args": dict(task.run_args),
            "held_out_subject": row.get("held_out_subject") or task.run_args.get("held_out_subject"),
            "train_subjects": row.get("train_subjects"),
            "val_subjects": row.get("val_subjects"),
            "test_subjects": row.get("test_subjects"),
            "data_ratio": row.get("data_ratio") or task.run_args.get("data_ratio"),
            "channel_variant": row.get("channel_variant") or task.run_args.get("channel_variant"),
            "channel_sparsity_config": row.get("channel_sparsity_config") or task.run_args.get("channel_sparsity_config"),
            "channel_count_target": row.get("channel_count_target") or task.run_args.get("channel_count_target"),
            "channel_count_loaded": row.get("channel_count_loaded"),
            "train_samples_original": row.get("train_samples_original"),
            "train_samples": row.get("train_samples"),
            "val_samples_original": row.get("val_samples_original"),
            "val_samples": row.get("val_samples"),
            "test_samples_original": row.get("test_samples_original"),
            "test_samples": row.get("test_samples"),
            "exp_id": row.get("exp_id"),
            "run_id": row.get("run_id"),
            "config_id": row.get("config_id"),
            "data_config_id": row.get("data_config_id"),
            "data_config_key": row.get("data_config_key"),
            "task": row.get("task"),
            "epoch_method": row.get("epoch_method"),
            "pipeline_key": row.get("pipeline_key"),
            "cache_root": row.get("cache_root"),
            "locator_cache_id": row.get("locator_cache_id"),
            "hdf5_cache_id": row.get("hdf5_cache_id"),
            "best_epoch": row.get("best_epoch"),
            "selection_metric_name": row.get("selection_metric_name"),
            "selection_metric_value": row.get("selection_metric_value"),
            "final_acc": row.get("final_acc"),
            "final_f1_macro": row.get("final_f1_macro"),
            "final_f1_weighted": row.get("final_f1_weighted"),
            "final_kappa": row.get("final_kappa"),
            "gpu": row.get("gpu"),
            "pid": row.get("pid"),
            "return_code": row.get("return_code"),
            "started_at": row.get("started_at"),
            "finished_at": row.get("finished_at"),
            "duration_sec": row.get("duration_sec"),
            "launcher_log": row.get("launcher_log"),
            "command": None,
            "carried_from_resume": True,
        }

    def _apply_resume_plan(self):
        self.prior_manifest_tasks = self._load_resume_manifest()
        self.prior_task_rows = self._load_resume_task_rows()
        current_manifest_tasks = self._index_manifest_tasks(
            self._current_manifest_tasks(),
            "current suite definition",
        )

        tasks_to_run = []
        skipped_success = 0
        kept_failed = 0
        retried_failed = 0
        resumed_unfinished = 0
        rerun_success = 0
        rerun_changed = 0
        no_history = 0

        for task in self.all_tasks:
            current_payload = current_manifest_tasks[self._task_identity_from_payload(task.manifest_payload())]
            job_name = current_payload["job_name"]
            identity = self._task_identity_from_payload(current_payload)
            prior_payload = self.prior_manifest_tasks.get(identity)
            prior_job_name = prior_payload.get("job_name") if prior_payload else None
            prior_row = self.prior_task_rows.get(prior_job_name or job_name)
            prior_status = prior_row.get("status") if prior_row else None
            self.prior_attempt_counts[job_name] = (prior_row or {}).get("attempt_count", 0)
            current_definition = self._task_definition_from_payload(
                current_payload,
                source_repo_root=self.repo_root,
            )
            definition_changed = (
                prior_payload is not None and
                self._task_definition_from_payload(
                    prior_payload,
                    source_repo_root=self.prior_manifest_repo_root,
                ) != current_definition
            )

            if definition_changed:
                tasks_to_run.append(task)
                self.resume_actions_by_job[job_name] = "rerun_changed"
                rerun_changed += 1
                continue

            if prior_status == "Success" and self.args.skip_completed:
                self.carried_completed_tasks.append(self._record_from_resume_row(task, prior_row))
                self.resume_actions_by_job[job_name] = "skip_success"
                skipped_success += 1
                continue

            if prior_status == "Failed" and not self.args.retry_failed:
                self.carried_failed_tasks.append(self._record_from_resume_row(task, prior_row))
                self.resume_actions_by_job[job_name] = "keep_failed"
                kept_failed += 1
                continue

            tasks_to_run.append(task)
            if prior_status == "Failed":
                self.resume_actions_by_job[job_name] = "retry_failed"
                retried_failed += 1
            elif prior_status in {"Pending", "Running"}:
                self.resume_actions_by_job[job_name] = "resume_unfinished"
                resumed_unfinished += 1
            elif prior_status == "Success":
                self.resume_actions_by_job[job_name] = "rerun_success"
                rerun_success += 1
            else:
                self.resume_actions_by_job[job_name] = "run_no_history"
                no_history += 1

        self.pending_tasks = deque(self._order_tasks_for_launch(tasks_to_run))
        self.resume_metadata.update(
            {
                "manifest_path": str(self.task_manifest_path),
                "task_status_csv_path": str(self.task_status_csv_path),
                "skipped_success_count": skipped_success,
                "kept_failed_count": kept_failed,
                "retried_failed_count": retried_failed,
                "resumed_unfinished_count": resumed_unfinished,
                "rerun_success_count": rerun_success,
                "rerun_changed_count": rerun_changed,
                "run_no_history_count": no_history,
                "scheduled_count": len(tasks_to_run),
            }
        )

    def _task_launch_group_key(self, task):
        return self._probe_cache_key(task.dataset, task.run_args)

    def _order_tasks_for_launch(self, tasks):
        ordered_tasks = list(tasks)
        if not getattr(self.args, "shuffle_tasks", False) or len(ordered_tasks) <= 1:
            return ordered_tasks

        rng = random.Random(self.args.shuffle_seed)
        grouped = {}
        for task in ordered_tasks:
            grouped.setdefault(self._task_launch_group_key(task), deque()).append(task)

        group_keys = list(grouped.keys())
        for key in group_keys:
            bucket = list(grouped[key])
            rng.shuffle(bucket)
            grouped[key] = deque(bucket)

        launch_order = []
        last_group_key = None
        while True:
            active_keys = [key for key in group_keys if grouped[key]]
            if not active_keys:
                break

            rng.shuffle(active_keys)
            if last_group_key is not None and len(active_keys) > 1 and active_keys[0] == last_group_key:
                for idx in range(1, len(active_keys)):
                    if active_keys[idx] != last_group_key:
                        active_keys[0], active_keys[idx] = active_keys[idx], active_keys[0]
                        break

            for key in active_keys:
                launch_order.append(grouped[key].popleft())
                last_group_key = key

        return launch_order

    def handle_stop(self, signum, _frame):
        self.stop_requested = True
        self.log(f"Received signal {signum}; stopping after active tasks finish.")

    def log(self, message):
        line = f"[{now_text()}] {message}"
        if not self._dashboard_enabled:
            print(line, flush=True)
        self.scheduler_log_file.write(line + "\n")
        self.scheduler_log_file.flush()

    def _preflight_notice(self, message):
        line = f"[{now_text()}] {message}"
        print(line, flush=True)
        self.scheduler_log_file.write(line + "\n")
        self.scheduler_log_file.flush()

    def console(self, message):
        line = f"[{now_text()}] {message}"
        print(line, flush=True)

    def console_block(self, lines):
        for line in lines:
            print(line, flush=True)

    def _setup_dashboard_console(self):
        if not self._dashboard_enabled or self._dashboard_cursor_hidden:
            return
        print("\033[?25l", end="", flush=True)
        self._dashboard_cursor_hidden = True

    def _teardown_dashboard_console(self):
        if not self._dashboard_enabled:
            return
        if self._dashboard_cursor_hidden:
            print("\033[?25h", end="", flush=True)
            self._dashboard_cursor_hidden = False

    def _render_console_snapshot(self, lines):
        if not self._dashboard_enabled:
            self.console_block(lines)
            return

        dashboard = "\n".join(lines)
        sys.stdout.write("\033[H\033[2J")
        sys.stdout.write(dashboard)
        if not dashboard.endswith("\n"):
            sys.stdout.write("\n")
        sys.stdout.flush()

    def _query_gpu_telemetry(self, emit_console=False):
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            error = "nvidia-smi not found"
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            stdout = (exc.stdout or "").strip()
            error = stderr or stdout or f"nvidia-smi exited with code {exc.returncode}"
        else:
            self._gpu_telemetry_error = None
            telemetry = {}
            for raw_line in result.stdout.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                parts = [part.strip() for part in line.split(",")]
                if len(parts) != 4:
                    continue
                gpu_index = parse_int_or_none(parts[0])
                if gpu_index is None or gpu_index not in self.args.gpus:
                    continue
                telemetry[gpu_index] = {
                    "memory_used_mib": parse_int_or_none(parts[1]),
                    "memory_total_mib": parse_int_or_none(parts[2]),
                    "utilization_gpu_percent": parse_int_or_none(parts[3]),
                }
            return telemetry

        if emit_console and error != self._gpu_telemetry_error:
            self.console(f"GPU telemetry unavailable: {error}")
        self._gpu_telemetry_error = error
        return {}

    def _build_gpu_snapshot(self, telemetry=None):
        telemetry = telemetry or {}
        snapshot = []
        for gpu_id, slot_index in self.gpu_slots:
            active_info = self.active.get((gpu_id, slot_index))
            gpu_entry = {
                "gpu": gpu_id,
                "slot": slot_index,
                "gpu_slot": self._format_gpu_slot(gpu_id, slot_index),
                "memory_used_mib": None,
                "memory_total_mib": None,
                "utilization_gpu_percent": None,
                "state": "idle",
                "pid": None,
                "job_name": None,
                "model": None,
                "dataset": None,
                "running_for_sec": None,
            }
            gpu_entry.update(telemetry.get(gpu_id, {}))

            if active_info is not None:
                task = active_info["task"]
                gpu_entry.update(
                    {
                        "state": "running",
                        "pid": active_info["proc"].pid,
                        "job_name": task.job_name,
                        "model": task.model,
                        "dataset": task.dataset,
                        "running_for_sec": int(time.time() - active_info["started_ts"]),
                    }
                )
            snapshot.append(gpu_entry)
        return snapshot

    def _progress_counts(self):
        completed = len(self._all_completed_records())
        failed = len(self._all_failed_records())
        finished = completed + failed
        percent = (finished / self.total_tasks * 100.0) if self.total_tasks else 100.0
        return {
            "total": self.total_tasks,
            "finished": finished,
            "completed": completed,
            "failed": failed,
            "active": len(self.active),
            "pending": len(self.pending_tasks),
            "percent": percent,
        }

    def _task_column_width(self):
        terminal_width = shutil.get_terminal_size(fallback=(160, 24)).columns
        fixed_columns = [
            ("GPU", 7),
            ("State", 7),
            ("PID", 8),
            ("Mem(MiB)", 15),
            ("Util", 6),
            ("Run", 8),
        ]
        border_width = 1 + sum(width + 3 for _, width in fixed_columns) + 1
        task_width = max(24, terminal_width - border_width - 3)
        return min(task_width, 120)

    def log_progress_snapshot(self):
        telemetry = self._query_gpu_telemetry(emit_console=True)
        gpu_snapshot = self._build_gpu_snapshot(telemetry=telemetry)
        progress = self._progress_counts()
        lines = [
            f"EEG-Arena Scheduler Dashboard | Updated {now_text()}",
            (
                "Progress: "
                f"{progress['finished']}/{progress['total']} ({progress['percent']:.1f}%) | "
                f"completed={progress['completed']} failed={progress['failed']} "
                f"active={progress['active']} pending={progress['pending']}"
            ),
        ]
        headers = [
            ("GPU", 7),
            ("State", 7),
            ("PID", 8),
            ("Mem(MiB)", 15),
            ("Util", 6),
            ("Run", 8),
            ("Task", self._task_column_width()),
        ]

        def border():
            return "+" + "+".join("-" * (width + 2) for _, width in headers) + "+"

        def row(values):
            cells = []
            for value, (_, width) in zip(values, headers):
                cells.append(f" {truncate_text(value, width):<{width}} ")
            return "|" + "|".join(cells) + "|"

        lines.extend([border(), row([label for label, _ in headers]), border()])
        for gpu_entry in gpu_snapshot:
            memory_used = gpu_entry["memory_used_mib"]
            memory_total = gpu_entry["memory_total_mib"]
            util = gpu_entry["utilization_gpu_percent"]
            memory_text = (
                f"{memory_used}/{memory_total}"
                if memory_used is not None and memory_total is not None
                else "N/A"
            )
            util_text = f"{util}%" if util is not None else "N/A"
            runtime_text = (
                format_duration_brief(gpu_entry["running_for_sec"])
                if gpu_entry["state"] == "running"
                else "-"
            )
            task_text = gpu_entry["job_name"] if gpu_entry["job_name"] else "-"
            pid_text = str(gpu_entry["pid"]) if gpu_entry["pid"] is not None else "-"
            lines.append(
                row(
                    [
                        gpu_entry["gpu_slot"],
                        gpu_entry["state"],
                        pid_text,
                        memory_text,
                        util_text,
                        runtime_text,
                        task_text,
                    ]
                )
            )
        lines.append(border())
        if self._gpu_telemetry_error:
            lines.append(f"Telemetry: unavailable ({self._gpu_telemetry_error})")
        self._render_console_snapshot(lines)

    def build_tasks(self):
        tasks = []
        task_index = 1

        if self.suite_spec.explicit_tasks:
            for explicit_task in self.suite_spec.explicit_tasks:
                tasks.append(
                    Task(
                        index=task_index,
                        model=explicit_task.model,
                        dataset=explicit_task.dataset,
                        run_args=dict(explicit_task.run_args),
                        explicit_name=explicit_task.name,
                    )
                )
                task_index += 1
            return tasks

        for dataset in self.suite_spec.datasets:
            for model in self.suite_spec.models:
                for sweep_args in iter_sweep_arg_maps(self.suite_spec, model=model, dataset=dataset):
                    base_run_args = resolve_task_run_args(
                        self.suite_spec,
                        model,
                        dataset,
                        sweep_args=sweep_args,
                    )
                    for run_args in self._expand_run_args_for_split_plan(dataset, base_run_args):
                        tasks.append(
                            Task(
                                index=task_index,
                                model=model,
                                dataset=dataset,
                                run_args=run_args,
                                explicit_name=None,
                            )
                        )
                        task_index += 1
        return tasks

    def _iter_matrix_base_runs(self):
        for dataset in self.suite_spec.datasets:
            for model in self.suite_spec.models:
                for sweep_args in iter_sweep_arg_maps(self.suite_spec, model=model, dataset=dataset):
                    yield dataset, model, resolve_task_run_args(
                        self.suite_spec,
                        model,
                        dataset,
                        sweep_args=sweep_args,
                    )

    def _probe_cache_key(self, dataset, run_args):
        dataset_name, task_name = split_dataset_task(dataset)
        return (
            dataset_name,
            task_name,
            clean_optional_text(run_args.get("cache_dir")),
            clean_optional_text(run_args.get("pipeline")),
            stable_json_text(run_args.get("pipeline_params")),
            stable_json_text(run_args.get("epoch_params")),
            bool(run_args.get("rebuild_hdf5", False)),
        )

    def _prepare_split_probe_plan(self):
        split_plan = self.suite_spec.split_plan or {}
        if "loso" not in split_plan:
            return

        seen = set()
        for dataset, _model, run_args in self._iter_matrix_base_runs():
            cache_key = self._probe_cache_key(dataset, run_args)
            if cache_key in seen:
                continue
            seen.add(cache_key)
            dataset_name, task_name = split_dataset_task(dataset)
            pipeline = clean_optional_text(run_args.get("pipeline"))
            self._split_probe_order.append(cache_key)
            self._split_probe_descriptors[cache_key] = {
                "dataset": dataset_name,
                "task": task_name,
                "pipeline": os.path.basename(pipeline) if pipeline else None,
            }

    @staticmethod
    def _build_subject_probe_params(run_args):
        from data.raw_to_hdf5 import HDF5BuildParams

        return HDF5BuildParams(
            epoch_params=dict(run_args.get("epoch_params") or {}) or None,
            pipeline=run_args.get("pipeline"),
            pipeline_params=dict(run_args.get("pipeline_params") or {}) or None,
            rebuild_hdf5=False,
            num_workers=run_args.get("num_workers") or 16,
        )

    def _get_loso_subject_ids(self, dataset, run_args):
        dataset_name, task_name = split_dataset_task(dataset)
        cache_key = self._probe_cache_key(dataset, run_args)
        if cache_key not in self._loso_subject_ids_cache:
            descriptor = self._split_probe_descriptors.get(cache_key, {})
            probe_index = self._split_probe_order.index(cache_key) + 1 if cache_key in self._split_probe_order else 1
            probe_total = len(self._split_probe_order) or 1
            pipeline_suffix = f", pipeline={descriptor['pipeline']}" if descriptor.get("pipeline") else ""
            self._preflight_notice(
                f"[split-plan {probe_index}/{probe_total}] Resolving LOSO subjects for "
                f"{descriptor.get('dataset', dataset_name)}:{descriptor.get('task', task_name)}{pipeline_suffix}"
            )
            buffer = StringIO()
            try:
                with redirect_stdout(buffer), redirect_stderr(buffer):
                    self._loso_subject_ids_cache[cache_key] = get_subject_ids(
                        dataset_name,
                        task_name,
                        build_params=self._build_subject_probe_params(run_args),
                        cache_dir=run_args.get("cache_dir"),
                    )
            except Exception as exc:
                self._preflight_notice(
                    f"[split-plan {probe_index}/{probe_total}] FAILED while resolving "
                    f"{descriptor.get('dataset', dataset_name)}:{descriptor.get('task', task_name)}: {exc}"
                )
                raise
            self._preflight_notice(
                f"[split-plan {probe_index}/{probe_total}] Resolved "
                f"{descriptor.get('dataset', dataset_name)}:{descriptor.get('task', task_name)} "
                f"-> {len(self._loso_subject_ids_cache[cache_key])} subjects"
            )
        return list(self._loso_subject_ids_cache[cache_key])

    def _expand_run_args_for_split_plan(self, dataset, run_args):
        split_plan = self.suite_spec.split_plan or {}
        if not split_plan:
            yield dict(run_args)
            return

        for split_mode, split_cfg in split_plan.items():
            if split_mode in {"cs", "ws", "cross_trial"}:
                folds = range(1, 6) if split_cfg.get("folds") == "all" else split_cfg.get("folds", [])
                for fold in folds:
                    expanded = dict(run_args)
                    expanded["split_mode"] = split_mode
                    expanded["fold"] = int(fold)
                    expanded.pop("held_out_subject", None)
                    yield expanded
                continue

            subject_ids = self._get_loso_subject_ids(dataset, run_args)
            subjects = subject_ids if split_cfg.get("subjects") == "all" else split_cfg.get("subjects", [])
            for subject in subjects:
                resolved_fold, resolved_subject = resolve_loso_target(
                    subject_ids,
                    fold=run_args.get("fold", 1),
                    held_out_subject=subject,
                )
                expanded = dict(run_args)
                expanded["split_mode"] = "loso"
                expanded["fold"] = resolved_fold
                expanded["held_out_subject"] = resolved_subject
                yield expanded

    def _extract_run_identifiers_from_log(self, launcher_log_path):
        identifiers = {
            "exp_id": None,
            "run_id": None,
            "config_id": None,
        }

        try:
            with Path(launcher_log_path).open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    exp_id_match = _EXPERIMENT_ID_PATTERN.match(stripped)
                    if exp_id_match:
                        identifiers["exp_id"] = exp_id_match.group(1).strip()
                        continue
                    run_id_match = _RUN_ID_PATTERN.match(stripped)
                    if run_id_match:
                        identifiers["run_id"] = run_id_match.group(1).strip()
                        continue
                    config_id_match = _CONFIG_ID_PATTERN.match(stripped)
                    if config_id_match:
                        identifiers["config_id"] = config_id_match.group(1).strip()
                        continue
        except OSError:
            pass

        return identifiers

    def _load_status_payload(self, run_id):
        run_id = clean_optional_text(run_id)
        if run_id is None:
            return None

        status_path = self.output_dir / "runs" / run_id / "status.json"
        try:
            with status_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            payload = None
        self._run_status_cache[run_id] = payload
        return payload

    def _resolve_cache_fields_for_run(self, run_id):
        status_payload = self._load_status_payload(run_id)
        if not isinstance(status_payload, dict):
            return {
                "cache_root": None,
                "locator_cache_id": None,
                "hdf5_cache_id": None,
            }
        return resolve_cache_link_fields(status_payload.get("cache_usage") or {})

    def _load_final_test_metrics_payload(self, run_id):
        run_id = clean_optional_text(run_id)
        if run_id is None:
            return None

        metrics_path = self.output_dir / "runs" / run_id / "metrics" / "final_test_metrics.json"
        try:
            with metrics_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            payload = None
        self._run_final_metrics_cache[run_id] = payload
        return payload

    def _relative_to_output_dir(self, path):
        text = clean_optional_text(path)
        if text is None:
            return None
        try:
            return str(Path(text).resolve().relative_to(self.output_dir))
        except ValueError:
            return None

    def _build_run_relpaths(self, run_id):
        run_id = clean_optional_text(run_id)
        if run_id is None:
            return {
                "run_relpath": None,
                "config_relpath": None,
                "status_relpath": None,
                "final_test_metrics_relpath": None,
                "log_relpath": None,
            }

        run_dir = Path("runs") / run_id
        return {
            "run_relpath": str(run_dir),
            "config_relpath": str(run_dir / "config.json"),
            "status_relpath": str(run_dir / "status.json"),
            "final_test_metrics_relpath": str(run_dir / "metrics" / "final_test_metrics.json"),
            "log_relpath": str(run_dir / "logs" / "console.log"),
        }

    def _resolve_run_detail_fields(self, run_id, *, fallback_record=None, dataset=None, task=None, run_args=None):
        fallback_record = fallback_record or {}
        run_args = run_args or {}
        status_payload = self._load_status_payload(run_id)
        final_test_payload = self._load_final_test_metrics_payload(run_id)
        cache_usage = {}
        if isinstance(status_payload, dict):
            cache_usage = status_payload.get("cache_usage") or {}
        sample_counts = (status_payload or {}).get("sample_counts") or {}

        fallback_dataset = clean_optional_text(dataset) or clean_optional_text(fallback_record.get("dataset"))
        fallback_task = clean_optional_text(task) or clean_optional_text(fallback_record.get("task"))

        pipeline_params = cache_usage.get("pipeline_params")
        if pipeline_params is None:
            pipeline_params = run_args.get("pipeline_params")
        epoch_params = cache_usage.get("epoch_params")
        if epoch_params is None:
            epoch_params = run_args.get("epoch_params")
        epoch_method = clean_optional_text(cache_usage.get("epoch_method")) or clean_optional_text(fallback_record.get("epoch_method"))
        pipeline_key = clean_optional_text(cache_usage.get("pipeline_key")) or clean_optional_text(fallback_record.get("pipeline_key"))
        data_config_fields = resolve_data_config_fields(
            dataset=cache_usage.get("dataset") or fallback_dataset,
            task=cache_usage.get("task") or fallback_task,
            epoch_method=epoch_method,
            pipeline_key=pipeline_key,
            pipeline_params=pipeline_params,
            epoch_params=epoch_params,
            data_config_key=cache_usage.get("data_config_key") or ((cache_usage.get("hdf5") or {}).get("config_key")) or fallback_record.get("data_config_key"),
            data_config_id=cache_usage.get("data_config_id") or fallback_record.get("data_config_id"),
        )
        cache_fields = resolve_cache_link_fields(cache_usage)
        if not any(cache_fields.values()):
            cache_fields = {
                "cache_root": clean_optional_text(fallback_record.get("cache_root")),
                "locator_cache_id": clean_optional_text(fallback_record.get("locator_cache_id")),
                "hdf5_cache_id": clean_optional_text(fallback_record.get("hdf5_cache_id")),
            }

        final_metric_fields = extract_final_metric_fields(final_test_payload)
        relpaths = self._build_run_relpaths(run_id)

        return {
            "task": clean_optional_text(cache_usage.get("task")) or fallback_task,
            "epoch_method": epoch_method,
            "pipeline_key": pipeline_key,
            "channel_variant": (
                clean_optional_text((status_payload or {}).get("channel_variant"))
                or clean_optional_text(run_args.get("channel_variant"))
                or clean_optional_text(fallback_record.get("channel_variant"))
            ),
            "channel_sparsity_config": (
                clean_optional_text((status_payload or {}).get("channel_sparsity_config"))
                or clean_optional_text(run_args.get("channel_sparsity_config"))
                or clean_optional_text(fallback_record.get("channel_sparsity_config"))
            ),
            "channel_count_target": (
                parse_int_or_none((status_payload or {}).get("channel_count_target"))
                or parse_int_or_none(run_args.get("channel_count_target"))
                or parse_int_or_none(fallback_record.get("channel_count_target"))
            ),
            "channel_count_loaded": (
                parse_int_or_none((status_payload or {}).get("channel_count_loaded"))
                or parse_int_or_none(fallback_record.get("channel_count_loaded"))
            ),
            "held_out_subject": (
                clean_optional_text((status_payload or {}).get("held_out_subject"))
                or clean_optional_text(run_args.get("held_out_subject"))
                or clean_optional_text(fallback_record.get("held_out_subject"))
            ),
            "train_subjects": (
                subject_list_text((status_payload or {}).get("train_subjects"))
                or subject_list_text(fallback_record.get("train_subjects"))
            ),
            "val_subjects": (
                subject_list_text((status_payload or {}).get("val_subjects"))
                or subject_list_text(fallback_record.get("val_subjects"))
            ),
            "test_subjects": (
                subject_list_text((status_payload or {}).get("test_subjects"))
                or subject_list_text(fallback_record.get("test_subjects"))
            ),
            "data_ratio": (
                (status_payload or {}).get("data_ratio")
                or run_args.get("data_ratio")
                or fallback_record.get("data_ratio")
            ),
            "train_samples_original": (
                parse_int_or_none((status_payload or {}).get("train_samples_original"))
                or parse_int_or_none(sample_counts.get("train_original"))
                or parse_int_or_none(fallback_record.get("train_samples_original"))
            ),
            "train_samples": (
                parse_int_or_none((status_payload or {}).get("train_samples"))
                or parse_int_or_none(sample_counts.get("train"))
                or parse_int_or_none(fallback_record.get("train_samples"))
            ),
            "val_samples_original": (
                parse_int_or_none((status_payload or {}).get("val_samples_original"))
                or parse_int_or_none(sample_counts.get("val_original"))
                or parse_int_or_none(fallback_record.get("val_samples_original"))
            ),
            "val_samples": (
                parse_int_or_none((status_payload or {}).get("val_samples"))
                or parse_int_or_none(sample_counts.get("val"))
                or parse_int_or_none(fallback_record.get("val_samples"))
            ),
            "test_samples_original": (
                parse_int_or_none((status_payload or {}).get("test_samples_original"))
                or parse_int_or_none(sample_counts.get("test_original"))
                or parse_int_or_none(fallback_record.get("test_samples_original"))
            ),
            "test_samples": (
                parse_int_or_none((status_payload or {}).get("test_samples"))
                or parse_int_or_none(sample_counts.get("test"))
                or parse_int_or_none(fallback_record.get("test_samples"))
            ),
            "cache_root": cache_fields.get("cache_root"),
            "locator_cache_id": cache_fields.get("locator_cache_id"),
            "hdf5_cache_id": cache_fields.get("hdf5_cache_id"),
            "best_epoch": parse_int_or_none((status_payload or {}).get("best_epoch")) or parse_int_or_none(fallback_record.get("best_epoch")),
            "selection_metric_name": clean_optional_text((status_payload or {}).get("selection_metric_name")) or clean_optional_text(fallback_record.get("selection_metric_name")),
            "selection_metric_value": (
                (status_payload or {}).get("selection_metric_value")
                if isinstance((status_payload or {}).get("selection_metric_value"), (int, float))
                else parse_float_or_none(fallback_record.get("selection_metric_value"))
            ),
            **data_config_fields,
            **final_metric_fields,
            **relpaths,
        }

    def _build_task_status_rows(self):
        latest_completed = {
            record.get("job_name"): record
            for record in self._all_completed_records()
            if record.get("job_name")
        }
        latest_failed = {
            record.get("job_name"): record
            for record in self._all_failed_records()
            if record.get("job_name")
        }
        attempts_by_job = {
            task.job_name: self.prior_attempt_counts.get(task.job_name, 0)
            for task in self.all_tasks
        }

        for record in self.failed_tasks:
            job_name = record.get("job_name")
            if not job_name:
                continue
            latest_failed[job_name] = record
            attempts_by_job[job_name] = attempts_by_job.get(job_name, 0) + 1

        for record in self.completed_tasks:
            job_name = record.get("job_name")
            if not job_name:
                continue
            latest_completed[job_name] = record
            attempts_by_job[job_name] = attempts_by_job.get(job_name, 0) + 1

        active_by_job = {}
        for (_gpu_id, _slot_index), info in self.active.items():
            job_name = info["task"].job_name
            active_by_job[job_name] = (info["gpu_id"], info)
            attempts_by_job[job_name] = attempts_by_job.get(job_name, 0) + 1

        rows = []
        for task in self.all_tasks:
            run_args = task.run_args or {}
            dataset_name, task_name = split_dataset_task(task.dataset)
            row = {
                "task_index": task.index,
                "job_name": task.job_name,
                "status": "Pending",
                "attempt_count": attempts_by_job.get(task.job_name, 0),
                "exp_id": None,
                "run_id": None,
                "config_id": None,
                "data_config_id": None,
                "data_config_key": None,
                "model": task.model,
                "dataset": dataset_name,
                "task": task_name,
                "split_mode": run_args.get("split_mode"),
                "held_out_subject": run_args.get("held_out_subject"),
                "fold": run_args.get("fold"),
                "train_subjects": None,
                "val_subjects": None,
                "test_subjects": None,
                "seed": run_args.get("seed"),
                "channel_system": run_args.get("channel_system"),
                "channel_variant": run_args.get("channel_variant"),
                "channel_sparsity_config": run_args.get("channel_sparsity_config"),
                "channel_count_target": run_args.get("channel_count_target"),
                "channel_count_loaded": None,
                "input_format": run_args.get("input_format"),
                "variant": run_args.get("variant"),
                "epochs": run_args.get("epochs"),
                "batch_size": run_args.get("batch_size"),
                "data_ratio": run_args.get("data_ratio", 1.0),
                "train_samples_original": None,
                "train_samples": None,
                "val_samples_original": None,
                "val_samples": None,
                "test_samples_original": None,
                "test_samples": None,
                "classifier_lr_scale": run_args.get("classifier_lr_scale"),
                "epoch_method": None,
                "pipeline_key": None,
                "pipeline": run_args.get("pipeline"),
                "pipeline_params": stable_json_text(run_args.get("pipeline_params")),
                "epoch_params": stable_json_text(run_args.get("epoch_params")),
                "cache_root": run_args.get("cache_dir"),
                "locator_cache_id": None,
                "hdf5_cache_id": None,
                "best_epoch": None,
                "selection_metric_name": None,
                "selection_metric_value": None,
                "final_acc": None,
                "final_f1_macro": None,
                "final_f1_weighted": None,
                "final_kappa": None,
                "started_at": None,
                "finished_at": None,
                "duration_sec": None,
                "gpu": None,
                "pid": None,
                "return_code": None,
                "run_relpath": None,
                "config_relpath": None,
                "status_relpath": None,
                "final_test_metrics_relpath": None,
                "log_relpath": None,
                "launcher_log": None,
            }

            completed_record = latest_completed.get(task.job_name)
            failed_record = latest_failed.get(task.job_name)
            active_entry = active_by_job.get(task.job_name)

            if active_entry is not None:
                gpu_id, info = active_entry
                active_identifiers = self._extract_run_identifiers_from_log(info["launcher_log"])
                detail_fields = self._resolve_run_detail_fields(
                    active_identifiers.get("run_id"),
                    fallback_record=info,
                    dataset=dataset_name,
                    task=task_name,
                    run_args=run_args,
                )
                row.update(
                    {
                        "status": "Running",
                        "exp_id": active_identifiers.get("exp_id"),
                        "run_id": active_identifiers.get("run_id"),
                        "config_id": active_identifiers.get("config_id"),
                        "gpu": gpu_id,
                        "pid": info["proc"].pid,
                        "started_at": info.get("started_at"),
                        "launcher_log": str(info.get("launcher_log")),
                        **detail_fields,
                    }
                )
            elif completed_record is not None:
                detail_fields = self._resolve_run_detail_fields(
                    completed_record.get("run_id"),
                    fallback_record=completed_record,
                    dataset=dataset_name,
                    task=task_name,
                    run_args=run_args,
                )
                row.update(
                    {
                        "status": "Success",
                        "exp_id": completed_record.get("exp_id"),
                        "run_id": completed_record.get("run_id"),
                        "config_id": completed_record.get("config_id"),
                        "gpu": completed_record.get("gpu"),
                        "pid": completed_record.get("pid"),
                        "return_code": completed_record.get("return_code"),
                        "started_at": completed_record.get("started_at"),
                        "finished_at": completed_record.get("finished_at"),
                        "duration_sec": completed_record.get("duration_sec"),
                        "launcher_log": completed_record.get("launcher_log"),
                        **detail_fields,
                    }
                )
            elif failed_record is not None:
                detail_fields = self._resolve_run_detail_fields(
                    failed_record.get("run_id"),
                    fallback_record=failed_record,
                    dataset=dataset_name,
                    task=task_name,
                    run_args=run_args,
                )
                row.update(
                    {
                        "status": "Failed",
                        "exp_id": failed_record.get("exp_id"),
                        "run_id": failed_record.get("run_id"),
                        "config_id": failed_record.get("config_id"),
                        "gpu": failed_record.get("gpu"),
                        "pid": failed_record.get("pid"),
                        "return_code": failed_record.get("return_code"),
                        "started_at": failed_record.get("started_at"),
                        "finished_at": failed_record.get("finished_at"),
                        "duration_sec": failed_record.get("duration_sec"),
                        "launcher_log": failed_record.get("launcher_log"),
                        **detail_fields,
                    }
                )

            rows.append(row)

        return rows

    def write_task_status_csv(self):
        rows = self._build_task_status_rows()
        self.task_status_csv_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.task_status_csv_path.with_suffix(self.task_status_csv_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TASK_STATUS_HEADERS)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        tmp_path.replace(self.task_status_csv_path)

    def write_manifest(self):
        task_payload = []
        for task in self.all_tasks:
            item = task.manifest_payload()
            item["resume_action"] = self.resume_actions_by_job.get(task.job_name, "run")
            task_payload.append(item)

        payload = {
            "created_at": now_text(),
            "repo_root": str(self.repo_root),
            "python_executable": sys.executable,
            "output_dir": str(self.output_dir),
            "scheduler_dir": str(self.scheduler_dir),
            "scheduler_log_path": str(self.scheduler_log_path),
            "suite_config_path": self.suite_spec.config_path,
            "suite_source": self.suite_spec.source,
            "suite_spec": suite_spec_to_dict(self.suite_spec),
            "sweeps": self.suite_spec.sweeps,
            "models": self.suite_spec.models,
            "datasets": self.suite_spec.datasets,
            "gpus": self.args.gpus,
            "processes_per_gpu": self.args.processes_per_gpu,
            "poll_interval": self.args.poll_interval,
            "resume": self.resume_metadata,
            "tasks": task_payload,
        }
        json_dump(self.task_manifest_path, payload)

    def write_state(self):
        telemetry = self._query_gpu_telemetry(emit_console=False)
        progress = self._progress_counts()
        gpu_snapshot = self._build_gpu_snapshot(telemetry=telemetry)
        payload = {
            "updated_at": now_text(),
            "repo_root": str(self.repo_root),
            "output_dir": str(self.output_dir),
            "scheduler_dir": str(self.scheduler_dir),
            "scheduler_log_path": str(self.scheduler_log_path),
            "launcher_log_dir": str(self.launcher_log_dir),
            "task_status_csv_path": str(self.task_status_csv_path),
            "resume": self.resume_metadata,
            "progress": progress,
            "pending_count": len(self.pending_tasks),
            "active_count": len(self.active),
            "completed_count": len(self._all_completed_records()),
            "failed_count": len(self._all_failed_records()),
            "pending": [task.job_name for task in self.pending_tasks],
            "pending_tasks": [
                {
                    "job_name": task.job_name,
                    "explicit_name": task.explicit_name,
                    "model": task.model,
                    "dataset": task.dataset,
                    "run_args": task.run_args,
                }
                for task in self.pending_tasks
            ],
            "active": [
                {
                    "gpu": info["gpu_id"],
                    "slot": info["slot_index"],
                    "gpu_slot": self._format_gpu_slot(info["gpu_id"], info["slot_index"]),
                    "pid": info["proc"].pid,
                    "job_name": info["task"].job_name,
                    "explicit_name": info["task"].explicit_name,
                    "model": info["task"].model,
                    "dataset": info["task"].dataset,
                    "run_args": info["task"].run_args,
                    "started_at": info["started_at"],
                    "launcher_log": str(info["launcher_log"]),
                }
                for _slot, info in sorted(self.active.items())
            ],
            "gpu_snapshot": gpu_snapshot,
            "gpu_telemetry_error": self._gpu_telemetry_error,
            "sweeps": self.suite_spec.sweeps,
        }
        json_dump(self.state_path, payload)
        self.write_task_status_csv()

    def build_command(self, task: Task):
        single_entry = self.repo_root / "run_single_experiment.py"
        if not single_entry.exists():
            single_entry = self.repo_root / "run_finetune.py"

        # task.dataset is stored as "dataset:task" — split for --dataset / --task args
        if ':' in task.dataset:
            dataset_name, task_name = task.dataset.split(':', 1)
        else:
            dataset_name = task.dataset
            task_name = None

        cmd = [
            sys.executable,
            "-u",
            str(single_entry),
            "--model",
            task.model,
            "--dataset",
            dataset_name,
        ]
        if task_name:
            cmd.extend(["--task", task_name])
        self._append_arg_map(cmd, task.run_args, keys=BASE_RUN_ARG_ORDER)
        cmd.extend(
            [
                "--output_dir",
                str(self.output_dir),
                "--device",
                "cuda:0",
                "--cuda",
                "0",
            ]
        )
        remaining_keys = [key for key in task.run_args.keys() if key not in BASE_RUN_ARG_ORDER]
        self._append_arg_map(cmd, task.run_args, keys=remaining_keys)
        return cmd

    def _append_arg_map(self, cmd, arg_map, keys):
        for key in keys:
            if key not in arg_map:
                continue
            self._append_single_arg(cmd, key, arg_map[key])

    @staticmethod
    def _append_single_arg(cmd, key, value):
        if value is None:
            return

        # Single-run entry and model configs currently register long options
        # using argparse dest-style names (underscores), so keep that form.
        if key in {"pipeline_params", "epoch_params"}:
            if isinstance(value, dict):
                items = [f"{param_key}={Scheduler._format_param_value(value[param_key])}" for param_key in sorted(value)]
            elif isinstance(value, (list, tuple)):
                items = [str(item) for item in value]
            else:
                raise ValueError(f"Unsupported value for '{key}': expected dict or list, got {type(value).__name__}.")
            if not items:
                return
            flag = "--pipeline_param" if key == "pipeline_params" else "--epoch_param"
            cmd.append(flag)
            cmd.extend(items)
            return

        flag = f"--{key}"
        if isinstance(value, bool):
            if key in BOOLEAN_VALUE_ARG_KEYS:
                cmd.extend([flag, "True" if value else "False"])
                return
            cmd.append(flag if value else f"--no-{key}")
            return

        cmd.extend([flag, str(value)])

    @staticmethod
    def _format_param_value(value):
        if isinstance(value, bool):
            return "True" if value else "False"
        return str(value)

    def launch_task(self, gpu_id: int, slot_index: int, task: Task):
        launcher_log = self.launcher_log_dir / f"{task.job_name}.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["PYTHONUNBUFFERED"] = "1"
        gpu_slot = self._format_gpu_slot(gpu_id, slot_index)

        cmd = self.build_command(task)
        log_file = launcher_log.open("w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            cwd=self.repo_root,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self.active[(gpu_id, slot_index)] = {
            "gpu_id": gpu_id,
            "slot_index": slot_index,
            "task": task,
            "proc": proc,
            "log_handle": log_file,
            "launcher_log": launcher_log,
            "cmd": cmd,
            "started_at": now_text(),
            "started_ts": time.time(),
        }
        self.log(
            f"GPU {gpu_slot}: launched {task.job_name} (pid={proc.pid}) -> {launcher_log}"
        )

    def reap_finished(self):
        finished_slots = []
        for slot, info in self.active.items():
            return_code = info["proc"].poll()
            if return_code is None:
                continue
            gpu_id, slot_index = slot
            gpu_slot = self._format_gpu_slot(gpu_id, slot_index)
            finished_slots.append(slot)
            info["log_handle"].close()
            task = info["task"]
            duration_sec = int(time.time() - info["started_ts"])
            run_identifiers = self._extract_run_identifiers_from_log(info["launcher_log"])
            cache_fields = self._resolve_cache_fields_for_run(run_identifiers.get("run_id"))
            record = {
                "job_name": task.job_name,
                "explicit_name": task.explicit_name,
                "model": task.model,
                "dataset": task.dataset,
                "run_args": task.run_args,
                "held_out_subject": task.run_args.get("held_out_subject"),
                "train_subjects": None,
                "val_subjects": None,
                "test_subjects": None,
                "data_ratio": task.run_args.get("data_ratio", 1.0),
                "channel_variant": task.run_args.get("channel_variant"),
                "channel_sparsity_config": task.run_args.get("channel_sparsity_config"),
                "channel_count_target": task.run_args.get("channel_count_target"),
                "channel_count_loaded": None,
                "train_samples_original": None,
                "train_samples": None,
                "val_samples_original": None,
                "val_samples": None,
                "test_samples_original": None,
                "test_samples": None,
                "exp_id": run_identifiers.get("exp_id"),
                "run_id": run_identifiers.get("run_id"),
                "config_id": run_identifiers.get("config_id"),
                "gpu": gpu_id,
                "gpu_slot": gpu_slot,
                "pid": info["proc"].pid,
                "return_code": return_code,
                "started_at": info["started_at"],
                "finished_at": now_text(),
                "duration_sec": duration_sec,
                "launcher_log": str(info["launcher_log"]),
                "command": info["cmd"],
                "cache_root": cache_fields.get("cache_root"),
                "locator_cache_id": cache_fields.get("locator_cache_id"),
                "hdf5_cache_id": cache_fields.get("hdf5_cache_id"),
            }
            if return_code == 0:
                self.completed_tasks.append(record)
                self.log(
                    f"GPU {gpu_slot}: completed {task.job_name} in {duration_sec}s"
                )
            else:
                self.failed_tasks.append(record)
                self.log(
                    f"GPU {gpu_slot}: FAILED {task.job_name} with rc={return_code}; see {info['launcher_log']}"
                )

        for slot in finished_slots:
            self.active.pop(slot, None)

        if finished_slots:
            self.write_state()

    def write_summary(self):
        payload = {
            "finished_at": now_text(),
            "output_dir": str(self.output_dir),
            "scheduler_dir": str(self.scheduler_dir),
            "scheduler_log_path": str(self.scheduler_log_path),
            "task_status_csv_path": str(self.task_status_csv_path),
            "resume": self.resume_metadata,
            "total_tasks": self.total_tasks,
            "completed_count": len(self._all_completed_records()),
            "failed_count": len(self._all_failed_records()),
            "completed": self._all_completed_records(),
            "failed": self._all_failed_records(),
        }
        json_dump(self.summary_path, payload)

    def run(self):
        try:
            self.log(
                f"Starting scheduler with {len(self.args.gpus)} GPUs, "
                f"{self.args.processes_per_gpu} process(es) per GPU "
                f"({len(self.gpu_slots)} total slots), {self.total_tasks} tasks, "
                f"output_dir={self.output_dir}"
            )
            if self.args.shuffle_tasks:
                self.log(
                    "Task launch shuffle enabled: cache-group-aware round-robin "
                    f"(seed={self.args.shuffle_seed})"
                )
            self.log(
                f"Models={','.join(self.suite_spec.models)} | Datasets={','.join(self.suite_spec.datasets)}"
            )
            self.log(f"Scheduler log: {self.scheduler_log_path}")
            if self.suite_spec.config_path:
                self.log(f"Suite config: {self.suite_spec.config_path}")
            if self.resume_metadata["enabled"]:
                self.log(
                    "Resume enabled: "
                    f"scheduled={self.resume_metadata.get('scheduled_count', 0)} "
                    f"skip_success={self.resume_metadata.get('skipped_success_count', 0)} "
                    f"keep_failed={self.resume_metadata.get('kept_failed_count', 0)} "
                    f"retry_failed={self.resume_metadata.get('retried_failed_count', 0)} "
                    f"resume_unfinished={self.resume_metadata.get('resumed_unfinished_count', 0)} "
                    f"rerun_changed={self.resume_metadata.get('rerun_changed_count', 0)} "
                    f"relaxed_host_paths={self.resume_metadata.get('relaxed_host_paths', False)} "
                    f"ignored_run_args={','.join(self.resume_metadata.get('ignored_run_args', [])) or 'none'}"
                )

            free_slots = deque(self.gpu_slots)

            while self.pending_tasks or self.active:
                launched_any = False
                while not self.stop_requested and self.pending_tasks and free_slots:
                    gpu_id, slot_index = free_slots.popleft()
                    task = self.pending_tasks.popleft()
                    self.launch_task(gpu_id, slot_index, task)
                    self.write_state()
                    launched_any = True

                if launched_any:
                    self.log_progress_snapshot()

                if not self.active:
                    break

                time.sleep(self.args.poll_interval)
                if self.active:
                    self.write_state()
                active_before = set(self.active.keys())
                self.reap_finished()
                active_after = set(self.active.keys())
                freed = sorted(active_before - active_after)
                for slot in freed:
                    free_slots.append(slot)
                if self.active or freed:
                    self.log_progress_snapshot()

            self.write_state()
            self.write_summary()
            self.log(
                f"Scheduler finished: completed={len(self._all_completed_records())}, "
                f"failed={len(self._all_failed_records())}, pending={len(self.pending_tasks)}"
            )
        finally:
            self._teardown_dashboard_console()
            self.scheduler_log_file.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Queue documented fine-tuning runs across multiple GPUs; see README.md."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional JSON suite config. When set, experiment design comes from this file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./experiments/output/simple_test",
        help="Shared output directory for the whole suite.",
    )
    parser.add_argument(
        "--scheduler-log",
        type=str,
        default=None,
        help="Optional scheduler log override. Defaults to <output_dir>/scheduler/scheduler.log.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reuse prior scheduler metadata under the same <output-dir>. Successful unchanged tasks are skipped by default; unfinished tasks and config-edited tasks are re-queued.",
    )
    parser.add_argument(
        "--retry-failed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only meaningful with --resume. Re-queue previously Failed tasks whose task definition has not changed.",
    )
    parser.add_argument(
        "--skip-completed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only meaningful with --resume. Skip tasks previously marked Success. Use --no-skip-completed to force reruns.",
    )
    parser.add_argument(
        "--resume-relaxed-host-paths",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only meaningful with --resume. Allow cross-host resume by ignoring relocatable cache paths and normalizing repo-local file paths such as --pipeline before checking whether a task definition changed.",
    )
    parser.add_argument(
        "--resume-ignore-run-args",
        type=str,
        nargs="+",
        default=[],
        metavar="KEY",
        help="Only meaningful with --resume. Ignore these run_args keys when checking whether a task definition changed. Useful for cross-device resume when keys like cache_dir move between hosts.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Training epochs per task.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size passed to run_single_experiment.py.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="DataLoader workers per task.",
    )
    parser.add_argument(
        "--data-ratio",
        "--data_ratio",
        dest="data_ratio",
        type=float,
        default=1.0,
        help="Randomly keep this fraction of each train/val/test split after splitting. Must be in (0, 1].",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed passed to run_single_experiment.py.",
    )
    parser.add_argument(
        "--split-mode",
        type=str,
        default="cs",
        help="Split mode passed to run_single_experiment.py (cs/ws/loso/cross_trial).",
    )
    parser.add_argument(
        "--held-out-subject",
        "--held_out_subject",
        dest="held_out_subject",
        type=str,
        default=None,
        help="Optional LOSO held-out subject passed to run_single_experiment.py.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=4,
        help="Fold passed to run_single_experiment.py (cross_trial uses fold=1).",
    )
    parser.add_argument(
        "--channel-system",
        type=str,
        default="10-20",
        choices=list(PUBLIC_CHANNEL_SYSTEM_CHOICES),
        help="Channel system passed to run_single_experiment.py.",
    )
    parser.add_argument(
        "--cache-dir",
        "--cache_dir",
        dest="cache_dir",
        type=str,
        default="./data/cache",
        help="Base HDF5/locator cache directory passed to run_single_experiment.py.",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=20,
        help="Seconds between child-process checks.",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4, 5, 6],
        help="GPU ids to use. By default, one task is launched per visible GPU at a time.",
    )
    parser.add_argument(
        "--processes-per-gpu",
        "--tasks-per-gpu",
        dest="processes_per_gpu",
        type=int,
        default=1,
        help="Maximum number of concurrent task processes to launch on each GPU.",
    )
    parser.add_argument(
        "--shuffle-tasks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Shuffle launch order with cache-group-aware round-robin so tasks sharing the same HDF5/cache key are spread apart.",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=0,
        help="Random seed used when --shuffle-tasks is enabled.",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=DEFAULT_MODEL_ORDER,
        choices=DEFAULT_MODEL_ORDER,
        help="Models to include in the suite.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=DEFAULT_DATASET_ORDER,
        choices=DEFAULT_DATASET_ORDER,
        help="Datasets to include in the suite.",
    )
    args = parser.parse_args(argv)
    if args.retry_failed and not args.resume:
        parser.error("--retry-failed requires --resume.")
    if not math.isfinite(args.data_ratio) or args.data_ratio <= 0.0 or args.data_ratio > 1.0:
        parser.error("--data-ratio must be in (0, 1].")
    if args.processes_per_gpu < 1:
        parser.error("--processes-per-gpu must be >= 1.")
    args._argv = list(sys.argv[1:] if argv is None else argv)
    args._resolved_suite_spec = resolve_suite_spec(args, parser=parser, argv=args._argv)
    return args


def main(argv=None):
    args = parse_args(argv)
    scheduler = Scheduler(args)
    scheduler.run()


if __name__ == "__main__":
    main()
