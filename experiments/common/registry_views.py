import csv
import os
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from experiments.common.result_schema import (
    CACHE_REGISTRY_HEADERS,
    CACHE_REGISTRY_VIEW_HEADERS,
    EXPERIMENTS_CSV_HEADERS,
    EXPERIMENTS_VIEW_HEADERS,
)
from experiments.common.runtime_metadata import clean_optional_path, clean_optional_text


INTERNAL_DIRNAME = "internal"
_INTERNAL_ONLY_FILENAMES = {
    "configs.jsonl",
    "cache_events.jsonl",
    "next_exp_id.txt",
    "next_exp_id.lock",
    "registry.lock",
    "cache_registry.lock",
    "views.lock",
}


@dataclass(frozen=True)
class RegistryLayout:
    repo_root: str
    registry_dir: str
    internal_dir: str
    experiments_view_path: str
    cache_registry_view_path: str
    experiments_canonical_path: str
    cache_registry_canonical_path: str
    configs_jsonl_path: str
    cache_events_path: str
    next_exp_id_path: str
    next_exp_id_lock_path: str
    registry_lock_path: str
    cache_registry_lock_path: str
    views_lock_path: str


def build_registry_layout(repo_root: str) -> RegistryLayout:
    repo_root = os.path.abspath(repo_root)
    registry_dir = os.path.join(repo_root, "experiments", "experiment_registry")
    internal_dir = os.path.join(registry_dir, INTERNAL_DIRNAME)
    return RegistryLayout(
        repo_root=repo_root,
        registry_dir=registry_dir,
        internal_dir=internal_dir,
        experiments_view_path=os.path.join(repo_root, "experiments.csv"),
        cache_registry_view_path=os.path.join(repo_root, "cache_registry.csv"),
        experiments_canonical_path=os.path.join(internal_dir, "experiments.canonical.csv"),
        cache_registry_canonical_path=os.path.join(internal_dir, "cache_registry.canonical.csv"),
        configs_jsonl_path=os.path.join(internal_dir, "configs.jsonl"),
        cache_events_path=os.path.join(internal_dir, "cache_events.jsonl"),
        next_exp_id_path=os.path.join(internal_dir, "next_exp_id.txt"),
        next_exp_id_lock_path=os.path.join(internal_dir, "next_exp_id.lock"),
        registry_lock_path=os.path.join(internal_dir, "registry.lock"),
        cache_registry_lock_path=os.path.join(internal_dir, "cache_registry.lock"),
        views_lock_path=os.path.join(internal_dir, "views.lock"),
    )


def ensure_registry_layout(layout: RegistryLayout) -> None:
    os.makedirs(layout.registry_dir, exist_ok=True)
    os.makedirs(layout.internal_dir, exist_ok=True)

    _maybe_move_root_csv(
        root_path=layout.experiments_view_path,
        internal_path=layout.experiments_canonical_path,
        expected_headers=EXPERIMENTS_CSV_HEADERS,
    )
    _maybe_move_root_csv(
        root_path=layout.cache_registry_view_path,
        internal_path=layout.cache_registry_canonical_path,
        expected_headers=CACHE_REGISTRY_HEADERS,
    )

    _move_or_remove_root_file(
        os.path.join(layout.registry_dir, "configs.jsonl"),
        layout.configs_jsonl_path,
    )
    _move_or_remove_root_file(
        os.path.join(layout.registry_dir, "cache_events.jsonl"),
        layout.cache_events_path,
    )
    _move_or_remove_root_file(
        os.path.join(layout.registry_dir, "next_exp_id.txt"),
        layout.next_exp_id_path,
    )
    _move_or_remove_root_file(
        os.path.join(layout.registry_dir, "next_exp_id.lock"),
        layout.next_exp_id_lock_path,
    )
    _move_or_remove_root_file(
        os.path.join(layout.registry_dir, "registry.lock"),
        layout.registry_lock_path,
    )
    _move_or_remove_root_file(
        os.path.join(layout.registry_dir, "cache_registry.lock"),
        layout.cache_registry_lock_path,
    )

    _move_or_remove_root_file(
        os.path.join(layout.registry_dir, "experiments.csv"),
        layout.experiments_view_path,
    )
    _move_or_remove_root_file(
        os.path.join(layout.registry_dir, "cache_registry.csv"),
        layout.cache_registry_view_path,
    )

    alias_path = os.path.join(layout.registry_dir, "path_aliases.csv")
    if os.path.exists(alias_path):
        os.remove(alias_path)


def refresh_registry_views(layout: RegistryLayout, repo_root: str) -> None:
    ensure_registry_layout(layout)
    with _gpfs_exclusive_lock(layout.views_lock_path):
        ensure_registry_layout(layout)
        with _canonical_read_locks(layout):
            experiments_rows, _ = _read_csv_rows(layout.experiments_canonical_path)
            cache_rows, _ = _read_csv_rows(layout.cache_registry_canonical_path)

        _write_csv_rows(
            layout.experiments_view_path,
            EXPERIMENTS_VIEW_HEADERS,
            _build_experiments_view_rows(
                experiments_rows=experiments_rows,
                cache_rows=cache_rows,
                repo_root=repo_root,
            ),
        )
        _write_csv_rows(
            layout.cache_registry_view_path,
            CACHE_REGISTRY_VIEW_HEADERS,
            build_cache_registry_view_rows(
                cache_rows=cache_rows,
                repo_root=repo_root,
            ),
        )


def build_cache_registry_view_rows(*, cache_rows, repo_root: str):
    projected_rows = []
    for row in cache_rows:
        cache_root = clean_optional_path(row.get("cache_root"))
        projected_rows.append(
            {
                "dataset": row.get("dataset"),
                "task": row.get("task"),
                "artifact_type": row.get("artifact_type"),
                "status": row.get("status"),
                "config_key": row.get("config_key"),
                "config_summary": _build_config_summary(row=row, repo_root=repo_root),
                "cache_root": _display_root_name(cache_root),
                "cache_path": _display_cache_path(
                    absolute_path=row.get("absolute_path"),
                    cache_root=cache_root,
                ),
                "size_human": _format_bytes(row.get("file_size_bytes")),
                "exists_on_disk": row.get("exists_on_disk"),
                "source_count": row.get("source_count"),
                "valid_source_count": row.get("valid_source_count"),
                "n_epochs_total": row.get("n_epochs_total"),
                "n_channels": row.get("n_channels"),
                "n_times": row.get("n_times"),
                "label_count": row.get("label_count"),
                "first_built_at": row.get("first_built_at"),
                "last_seen_at": row.get("last_seen_at"),
                "hit_count": row.get("hit_count"),
                "build_count": row.get("build_count"),
                "rebuild_count": row.get("rebuild_count"),
                "cache_id": row.get("cache_id"),
                "cache_base_id": row.get("cache_base_id"),
                "parent_locator_cache_id": row.get("parent_locator_cache_id"),
                "cache_key": _display_cache_key(
                    cache_key=row.get("cache_key"),
                    repo_root=repo_root,
                ),
                "producer_exp_id": row.get("producer_exp_id"),
                "producer_run_id": row.get("producer_run_id"),
                "last_exp_id": row.get("last_exp_id"),
                "last_run_id": row.get("last_run_id"),
            }
        )

    return projected_rows


def canonical_experiments_csv_path(registry_dir: str) -> str:
    registry_dir = os.path.abspath(registry_dir)
    return os.path.join(registry_dir, INTERNAL_DIRNAME, "experiments.canonical.csv")


def canonical_cache_registry_csv_path(registry_dir: str) -> str:
    registry_dir = os.path.abspath(registry_dir)
    return os.path.join(registry_dir, INTERNAL_DIRNAME, "cache_registry.canonical.csv")


def _read_csv_rows(path: str):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return [], []
    with open(path, "r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        return list(reader), list(reader.fieldnames or [])


def _write_csv_rows(path: str, headers, rows) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp",
        dir=os.path.dirname(path),
    )
    os.close(fd)
    try:
        with open(tmp_path, "w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=headers)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field) for field in headers})
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _maybe_move_root_csv(root_path: str, internal_path: str, expected_headers) -> None:
    if not os.path.exists(root_path):
        return
    if os.path.exists(internal_path):
        return

    _, headers = _read_csv_rows(root_path)
    if headers == list(expected_headers):
        os.replace(root_path, internal_path)


def _move_or_remove_root_file(root_path: str, internal_path: str) -> None:
    if not os.path.exists(root_path):
        return
    os.makedirs(os.path.dirname(internal_path), exist_ok=True)
    if not os.path.exists(internal_path):
        os.replace(root_path, internal_path)
        return
    os.remove(root_path)


@contextmanager
def _gpfs_exclusive_lock(lock_path: str, timeout: float = 900.0, stale_timeout: float = 7200.0):
    """Cross-node safe exclusive lock for GPFS using atomic O_CREAT|O_EXCL.

    fcntl.flock is unreliable across nodes on networked filesystems like GPFS.
    O_CREAT|O_EXCL is atomic on GPFS even across nodes.
    """
    xlock_path = lock_path + ".xlock"
    os.makedirs(os.path.dirname(xlock_path), exist_ok=True)
    deadline = time.monotonic() + timeout
    poll = 0.05
    while True:
        try:
            fd = os.open(xlock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(xlock_path) > stale_timeout:
                    os.unlink(xlock_path)
                    continue
            except FileNotFoundError:
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Could not acquire GPFS lock after {timeout}s: {xlock_path}")
            time.sleep(poll)
            poll = min(poll * 1.5, 2.0)
    try:
        yield
    finally:
        try:
            os.unlink(xlock_path)
        except FileNotFoundError:
            pass


@contextmanager
def _canonical_read_locks(layout: RegistryLayout):
    with _gpfs_exclusive_lock(layout.registry_lock_path):
        with _gpfs_exclusive_lock(layout.cache_registry_lock_path):
            yield


def _build_experiments_view_rows(*, experiments_rows, cache_rows, repo_root: str):
    cache_rows_by_id = {
        clean_optional_text(row.get("cache_id")): row
        for row in cache_rows
        if clean_optional_text(row.get("cache_id"))
    }

    projected_rows = []
    for row in experiments_rows:
        hdf5_row = cache_rows_by_id.get(clean_optional_text(row.get("hdf5_cache_id")))
        locator_row = cache_rows_by_id.get(clean_optional_text(row.get("locator_cache_id")))
        cache_row = hdf5_row or locator_row or {}
        cache_root = clean_optional_path(cache_row.get("cache_root")) or clean_optional_path(row.get("cache_root"))

        projected_rows.append(
            {
                "exp_id": row.get("exp_id"),
                "status": row.get("status"),
                "model": row.get("model"),
                "dataset": row.get("dataset"),
                "task": row.get("task"),
                "split_mode": row.get("split_mode"),
                "held_out_subject": row.get("held_out_subject"),
                "fold": row.get("fold"),
                "train_subjects": row.get("train_subjects"),
                "val_subjects": row.get("val_subjects"),
                "test_subjects": row.get("test_subjects"),
                "seed": row.get("seed"),
                "input_format": row.get("input_format"),
                "channel_system": row.get("channel_system"),
                "channel_variant": row.get("channel_variant"),
                "channel_sparsity_config": row.get("channel_sparsity_config"),
                "channel_count_target": row.get("channel_count_target"),
                "channel_count_loaded": row.get("channel_count_loaded"),
                "variant": row.get("variant"),
                "data_ratio": row.get("data_ratio"),
                "train_samples_original": row.get("train_samples_original"),
                "train_samples": row.get("train_samples"),
                "val_samples_original": row.get("val_samples_original"),
                "val_samples": row.get("val_samples"),
                "test_samples_original": row.get("test_samples_original"),
                "test_samples": row.get("test_samples"),
                "epoch_method": row.get("epoch_method"),
                "pipeline_name": _display_pipeline_name(
                    pipeline=row.get("pipeline"),
                    pipeline_key=row.get("pipeline_key"),
                    repo_root=repo_root,
                ),
                "pipeline_params": row.get("pipeline_params"),
                "epoch_params": row.get("epoch_params"),
                "data_config_key": row.get("data_config_key"),
                "suite_name": row.get("suite_name"),
                "cache_root": _display_root_name(cache_root),
                "cache_path": _display_cache_path(
                    absolute_path=(cache_row or {}).get("absolute_path"),
                    cache_root=cache_root,
                ),
                "final_acc": row.get("final_acc"),
                "final_f1_macro": row.get("final_f1_macro"),
                "final_f1_weighted": row.get("final_f1_weighted"),
                "final_kappa": row.get("final_kappa"),
                "best_epoch": row.get("best_epoch"),
                "selection_metric_name": row.get("selection_metric_name"),
                "selection_metric_value": row.get("selection_metric_value"),
                "started_at": row.get("started_at"),
                "finished_at": row.get("finished_at"),
                "duration_sec": row.get("duration_sec"),
                "run_id": row.get("run_id"),
                "config_id": row.get("config_id"),
                "data_config_id": row.get("data_config_id"),
                "locator_cache_id": row.get("locator_cache_id"),
                "hdf5_cache_id": row.get("hdf5_cache_id"),
            }
        )

    return projected_rows


def _build_config_summary(*, row: dict, repo_root: str) -> str | None:
    artifact_type = clean_optional_text(row.get("artifact_type"))
    if artifact_type == "locator":
        parts = [
            f"domain={clean_optional_text(row.get('domain_tag')) or 'none'}",
            f"kernel={_display_repo_or_short_path(row.get('kernel_spec'), repo_root) or 'none'}",
        ]
        return "; ".join(parts)

    pipeline_name = _display_pipeline_name(
        pipeline=row.get("pipeline"),
        pipeline_key=row.get("pipeline_key"),
        repo_root=repo_root,
    ) or "none"
    epoch_method = clean_optional_text(row.get("epoch_method")) or "none"
    pipeline_params = clean_optional_text(row.get("pipeline_params")) or "none"
    epoch_params = clean_optional_text(row.get("epoch_params")) or "none"
    return (
        f"epoch={epoch_method}; "
        f"pipeline={pipeline_name}; "
        f"pipeline_params={pipeline_params}; "
        f"epoch_params={epoch_params}"
    )


def _display_pipeline_name(*, pipeline, pipeline_key, repo_root: str) -> str | None:
    display = _display_repo_or_short_path(pipeline, repo_root)
    if display:
        return display
    return clean_optional_text(pipeline_key)


def _display_cache_key(*, cache_key, repo_root: str) -> str | None:
    cache_key = clean_optional_text(cache_key)
    if cache_key is None:
        return None

    parts = cache_key.split("::")
    if len(parts) != 4 or parts[0] != "locator":
        return cache_key

    kernel_display = _display_repo_or_short_path(parts[3], repo_root)
    if kernel_display is None:
        return cache_key
    return "::".join([parts[0], parts[1], parts[2], kernel_display])


def _display_repo_or_short_path(path_value, repo_root: str) -> str | None:
    path_value = clean_optional_text(path_value)
    if path_value is None:
        return None

    absolute_path = clean_optional_path(path_value)
    if absolute_path is None:
        return path_value

    repo_short = _display_repo_relative_path(absolute_path, repo_root)
    if repo_short is not None:
        return repo_short
    return _display_tail_path(absolute_path)


def _display_repo_relative_path(path_value: str, repo_root: str) -> str | None:
    absolute_path = clean_optional_path(path_value)
    repo_root = clean_optional_path(repo_root)
    if absolute_path is None or repo_root is None:
        return None
    try:
        relative = os.path.relpath(absolute_path, repo_root)
    except ValueError:
        return None
    if relative.startswith(".."):
        return None
    return relative.replace(os.sep, "/")


def _display_cache_path(*, absolute_path, cache_root) -> str | None:
    absolute_path = clean_optional_path(absolute_path)
    cache_root = clean_optional_path(cache_root)
    if absolute_path is None:
        return None

    if cache_root is not None:
        try:
            relative = os.path.relpath(absolute_path, cache_root)
        except ValueError:
            relative = None
        if relative == ".":
            return _display_root_name(cache_root)
        if relative and not relative.startswith(".."):
            return f"{_display_root_name(cache_root)}/{relative.replace(os.sep, '/')}"

    return _display_tail_path(absolute_path)


def _display_root_name(root_path: str | None) -> str | None:
    root_path = clean_optional_path(root_path)
    if root_path is None:
        return None
    name = os.path.basename(root_path.rstrip(os.sep))
    return name or root_path


def _display_tail_path(path_value: str, parts: int = 3) -> str | None:
    absolute_path = clean_optional_path(path_value)
    if absolute_path is None:
        return None
    path_parts = Path(absolute_path).parts
    if len(path_parts) <= parts:
        return absolute_path
    return "/".join(path_parts[-parts:])


def _format_bytes(value) -> str | None:
    text = clean_optional_text(value)
    if text is None:
        return None
    try:
        size = int(text)
    except (TypeError, ValueError):
        return None

    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    amount = float(size)
    unit_index = 0
    while amount >= 1024 and unit_index < len(units) - 1:
        amount /= 1024.0
        unit_index += 1

    if unit_index == 0:
        return f"{int(amount)} {units[unit_index]}"
    if amount >= 100:
        return f"{amount:.0f} {units[unit_index]}"
    if amount >= 10:
        return f"{amount:.1f} {units[unit_index]}"
    return f"{amount:.2f} {units[unit_index]}"
