import csv
import hashlib
import json
import math
import os
from datetime import datetime
from pathlib import Path

from experiments.common.registry_views import (
    _gpfs_exclusive_lock,
    build_cache_registry_view_rows,
    build_registry_layout,
    ensure_registry_layout,
    refresh_registry_views,
)
from experiments.common.result_schema import CACHE_REGISTRY_HEADERS, CACHE_REGISTRY_VIEW_HEADERS

_CACHE_REGISTRY_ID_COLUMNS = ("locator_cache_id", "hdf5_cache_id")
_CACHE_REGISTRY_PATH_FIELDS = {"cache_root", "raw_path", "kernel_spec", "absolute_path", "pipeline"}
_CACHE_REGISTRY_TEXT_FIELDS = {
    "cache_id",
    "cache_base_id",
    "cache_key",
    "artifact_type",
    "status",
    "dataset",
    "task",
    "parent_locator_cache_id",
    "domain_tag",
    "relative_path",
    "epoch_method",
    "config_key",
    "pipeline_key",
    "pipeline_params",
    "epoch_params",
    "last_error",
    "producer_run_id",
    "last_run_id",
}


def now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")


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
    if isinstance(value, set):
        return [normalize_for_json(item) for item in sorted(value)]
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return str(value)
    return value


def stable_json_text(value):
    if value is None:
        return None
    return json.dumps(
        normalize_for_json(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def clean_optional_text(value):
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def clean_optional_path(value):
    text = clean_optional_text(value)
    if text is None:
        return None
    return os.path.abspath(text)


def parse_int_or_none(value):
    text = clean_optional_text(value)
    if text is None:
        return None
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def parse_bool(value, default=False):
    if isinstance(value, bool):
        return value
    text = clean_optional_text(value)
    if text is None:
        return default
    return text.lower() == "true"


def _short_sha1(*parts: str, length: int) -> str:
    canonical = "||".join(parts)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:length]


def _slugify_task_name(value: str | None) -> str:
    text = (clean_optional_text(value) or "unknown").lower()
    chars = []
    previous_is_sep = False
    for char in text:
        if char.isalnum():
            chars.append(char)
            previous_is_sep = False
            continue
        if not previous_is_sep:
            chars.append("_")
        previous_is_sep = True
    slug = "".join(chars).strip("_")
    return slug or "unknown"


def extract_cache_base_id(cache_id: str | None) -> str | None:
    cache_id = clean_optional_text(cache_id)
    if cache_id is None:
        return None
    if cache_id.endswith(":locator"):
        return cache_id[: -len(":locator")]
    marker = ":hdf5:"
    if marker in cache_id:
        return cache_id.split(marker, 1)[0]
    return None


def build_cache_base_id(
    cache_root: str,
    dataset: str,
    raw_path: str | None,
    domain_tag: str | None,
    kernel_spec: str | None,
) -> str:
    return (
        "cache_"
        + _short_sha1(
            clean_optional_path(cache_root) or "",
            clean_optional_text(dataset) or "",
            clean_optional_path(raw_path) or "",
            clean_optional_text(domain_tag) or "",
            clean_optional_path(kernel_spec) or "",
            length=16,
        )
    )


def build_locator_cache_id(cache_base_id: str) -> str:
    return f"{cache_base_id}:locator"


def build_hdf5_cache_id(cache_base_id: str, task: str | None, config_key: str | None) -> str:
    task_slug = _slugify_task_name(task)
    config_hash = _short_sha1(clean_optional_text(config_key) or "", length=12)
    return f"{cache_base_id}:hdf5:{task_slug}:{config_hash}"


def build_locator_cache_key(dataset: str, domain_tag: str | None, kernel_spec: str | None) -> str:
    kernel_text = clean_optional_path(kernel_spec) or "none"
    domain_text = clean_optional_text(domain_tag) or "none"
    return f"locator::{dataset}::{domain_text}::{kernel_text}"


def build_hdf5_cache_key(dataset: str, task: str, config_key: str) -> str:
    return f"hdf5::{dataset}::{task}::{config_key}"


def cache_relative_path(cache_root: str, absolute_path: str) -> str:
    relative = os.path.relpath(os.path.abspath(absolute_path), os.path.abspath(cache_root))
    return relative.replace(os.sep, "/")


def read_locator_stats(locator_path: str) -> dict:
    stats = {
        "source_count": None,
        "valid_source_count": None,
    }
    if not os.path.exists(locator_path) or os.path.getsize(locator_path) == 0:
        return stats

    total = 0
    valid = 0
    with open(locator_path, "r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            total += 1
            completeness = clean_optional_text(row.get("Completeness Check"))
            if completeness == "Completed":
                valid += 1

    stats["source_count"] = total
    stats["valid_source_count"] = valid
    return stats


def read_hdf5_stats(hdf5_path: str) -> dict:
    import h5py

    stats = {
        "n_epochs_total": None,
        "n_channels": None,
        "n_times": None,
        "label_count": None,
    }
    if not os.path.exists(hdf5_path) or os.path.getsize(hdf5_path) == 0:
        return stats

    with h5py.File(hdf5_path, "r") as handle:
        data_ds = handle["data"]
        stats["n_epochs_total"] = int(handle.attrs.get("n_epochs_total", data_ds.shape[0]))
        stats["n_channels"] = int(handle.attrs.get("n_channels", data_ds.shape[1]))
        stats["n_times"] = int(handle.attrs.get("n_times", data_ds.shape[2]))
        label_map_raw = handle.attrs.get("label_map")
        if label_map_raw:
            if isinstance(label_map_raw, bytes):
                label_map_raw = label_map_raw.decode("utf-8")
            stats["label_count"] = len(json.loads(label_map_raw))
    return stats


class CacheRegistry:
    def __init__(self, repo_root: str):
        self.repo_root = os.path.abspath(repo_root)
        self.layout = build_registry_layout(self.repo_root)
        ensure_registry_layout(self.layout)
        self.registry_dir = self.layout.registry_dir
        self.cache_registry_path = self.layout.cache_registry_canonical_path
        self.cache_events_path = self.layout.cache_events_path
        self.cache_registry_lock_path = self.layout.cache_registry_lock_path
        self.experiments_csv_path = self.layout.experiments_canonical_path
        self.output_root = os.path.join(self.repo_root, "experiments", "output")
        self._migration_checked = False
        self._ensure_migrated()

    def record_event(
        self,
        *,
        row_update: dict,
        event_type: str,
        exp_id=None,
        run_id=None,
        error_message: str | None = None,
        update_registry: bool = True,
        is_rebuild: bool = False,
    ) -> dict | None:
        self._ensure_migrated()

        row_update = self._normalize_registry_row(row_update)
        cache_id = clean_optional_text(row_update.get("cache_id"))
        if cache_id is None:
            raise ValueError("row_update.cache_id is required for cache registry events.")

        os.makedirs(self.registry_dir, exist_ok=True)
        with _gpfs_exclusive_lock(self.cache_registry_lock_path):
            rows = self._load_rows_locked()
            rows_by_id = {row["cache_id"]: row for row in rows if clean_optional_text(row.get("cache_id"))}
            prior_row = rows_by_id.get(cache_id)

            if update_registry:
                merged_row = self._merge_row(
                    prior_row=prior_row,
                    row_update=row_update,
                    event_type=event_type,
                    exp_id=exp_id,
                    run_id=run_id,
                    error_message=error_message,
                    is_rebuild=is_rebuild,
                )
                rows_by_id[cache_id] = merged_row
                self._write_rows_locked(rows_by_id.values())
            else:
                merged_row = prior_row or self._seed_row(row_update)

            self._append_event_locked(
                row=merged_row,
                row_update=row_update,
                prior_row=prior_row,
                event_type=event_type,
                exp_id=exp_id,
                run_id=run_id,
                error_message=error_message,
            )

            cache_root = clean_optional_path((merged_row or {}).get("cache_root"))
            if update_registry and cache_root:
                self._safe_sync_local_registry_locked(cache_root, rows_by_id.values())

        if update_registry:
            refresh_registry_views(self.layout, self.repo_root)
        return merged_row

    def _is_registry_up_to_date(self) -> bool:
        """Fast path: check if cache registry already has the correct headers.

        This avoids acquiring the global lock when the registry is already
        in the latest format, which is the common case after the first run.
        """
        if not os.path.exists(self.cache_registry_path):
            return False
        try:
            if os.path.getsize(self.cache_registry_path) == 0:
                return False
            with open(self.cache_registry_path, "r", newline="", encoding="utf-8") as csv_file:
                reader = csv.reader(csv_file)
                headers = next(reader, None)
            return headers == list(CACHE_REGISTRY_HEADERS)
        except Exception:
            return False

    def _ensure_migrated(self) -> None:
        if self._migration_checked:
            return

        # Fast path: skip lock acquisition if registry is already up-to-date.
        if self._is_registry_up_to_date():
            self._migration_checked = True
            return

        os.makedirs(self.registry_dir, exist_ok=True)
        with _gpfs_exclusive_lock(self.cache_registry_lock_path):
            # Double-check after acquiring lock (another process may have migrated).
            if self._is_registry_up_to_date():
                self._migration_checked = True
                return
            self._migrate_storage_locked()
            self._migration_checked = True
        refresh_registry_views(self.layout, self.repo_root)

    def _migrate_storage_locked(self) -> None:
        rows, headers = self._load_csv_rows(self.cache_registry_path)
        migrated_rows, cache_id_map, cache_base_map, changed = self._migrate_registry_rows(rows, headers)

        if changed or (os.path.exists(self.cache_registry_path) and headers != CACHE_REGISTRY_HEADERS):
            self._write_csv_rows(self.cache_registry_path, CACHE_REGISTRY_HEADERS, migrated_rows)
        elif not os.path.exists(self.cache_registry_path):
            self._write_csv_rows(self.cache_registry_path, CACHE_REGISTRY_HEADERS, [])

        self._rewrite_cache_events_locked(cache_id_map, cache_base_map)
        self._rewrite_csv_id_columns_locked(self.experiments_csv_path, _CACHE_REGISTRY_ID_COLUMNS, cache_id_map)
        self._rewrite_task_status_files_locked(cache_id_map)
        self._rewrite_status_json_files_locked(cache_id_map, cache_base_map)

        for cache_root in sorted(
            {
                clean_optional_path(row.get("cache_root"))
                for row in migrated_rows
                if clean_optional_path(row.get("cache_root"))
            }
        ):
            self._safe_sync_local_registry_locked(cache_root, migrated_rows)

    def _migrate_registry_rows(self, rows, headers):
        changed = headers != CACHE_REGISTRY_HEADERS
        cache_id_map = {}
        cache_base_map = {}
        deduped_rows = {}

        for raw_row in rows:
            old_cache_id = clean_optional_text(raw_row.get("cache_id"))
            migrated = self._normalize_registry_row(raw_row)
            new_cache_id = clean_optional_text(migrated.get("cache_id"))
            new_cache_base_id = clean_optional_text(migrated.get("cache_base_id"))

            if old_cache_id and new_cache_id:
                cache_id_map[old_cache_id] = new_cache_id
            if old_cache_id and new_cache_base_id:
                cache_base_map[old_cache_id] = new_cache_base_id
            if new_cache_id and new_cache_base_id:
                cache_base_map[new_cache_id] = new_cache_base_id

            if old_cache_id != new_cache_id:
                changed = True
            if clean_optional_text(raw_row.get("cache_base_id")) != new_cache_base_id:
                changed = True
            if clean_optional_text(raw_row.get("parent_locator_cache_id")) != clean_optional_text(migrated.get("parent_locator_cache_id")):
                changed = True

            if new_cache_id is None:
                continue
            if new_cache_id in deduped_rows:
                changed = True
            deduped_rows[new_cache_id] = migrated

        migrated_rows = list(deduped_rows.values())
        if len(migrated_rows) != len(rows):
            changed = True
        return migrated_rows, cache_id_map, cache_base_map, changed

    def _normalize_registry_row(self, row: dict) -> dict:
        normalized = {field: row.get(field) for field in CACHE_REGISTRY_HEADERS}

        for field in _CACHE_REGISTRY_PATH_FIELDS:
            normalized[field] = clean_optional_path(normalized.get(field))
        for field in _CACHE_REGISTRY_TEXT_FIELDS:
            normalized[field] = clean_optional_text(normalized.get(field))

        artifact_type = clean_optional_text(normalized.get("artifact_type"))
        cache_base_id = build_cache_base_id(
            clean_optional_path(normalized.get("cache_root")) or "",
            clean_optional_text(normalized.get("dataset")) or "",
            clean_optional_path(normalized.get("raw_path")) or "",
            clean_optional_text(normalized.get("domain_tag")) or "",
            clean_optional_path(normalized.get("kernel_spec")) or "",
        )
        if not clean_optional_text(normalized.get("cache_base_id")):
            normalized["cache_base_id"] = cache_base_id
        else:
            normalized["cache_base_id"] = clean_optional_text(normalized.get("cache_base_id"))

        resolved_base_id = clean_optional_text(normalized.get("cache_base_id")) or extract_cache_base_id(normalized.get("cache_id"))
        if resolved_base_id:
            normalized["cache_base_id"] = resolved_base_id

        if artifact_type == "locator":
            normalized["cache_id"] = build_locator_cache_id(normalized["cache_base_id"])
            normalized["parent_locator_cache_id"] = None
            normalized["task"] = None
        elif artifact_type == "hdf5":
            normalized["cache_id"] = build_hdf5_cache_id(
                normalized["cache_base_id"],
                normalized.get("task"),
                normalized.get("config_key"),
            )
            normalized["parent_locator_cache_id"] = build_locator_cache_id(normalized["cache_base_id"])
        else:
            normalized["cache_id"] = clean_optional_text(normalized.get("cache_id"))
            normalized["parent_locator_cache_id"] = clean_optional_text(normalized.get("parent_locator_cache_id"))

        return normalized

    def _seed_row(self, row_update: dict) -> dict:
        row = {field: None for field in CACHE_REGISTRY_HEADERS}
        row.update(
            {
                "exists_on_disk": False,
                "build_count": 0,
                "rebuild_count": 0,
                "hit_count": 0,
            }
        )
        for field in CACHE_REGISTRY_HEADERS:
            if field in row_update and row_update[field] is not None:
                row[field] = row_update[field]
        return self._normalize_registry_row(row)

    def _merge_row(
        self,
        *,
        prior_row: dict | None,
        row_update: dict,
        event_type: str,
        exp_id,
        run_id,
        error_message: str | None,
        is_rebuild: bool,
    ) -> dict:
        timestamp = now_iso()
        merged = self._seed_row(prior_row or {})

        if prior_row:
            merged["build_count"] = parse_int_or_none(prior_row.get("build_count")) or 0
            merged["rebuild_count"] = parse_int_or_none(prior_row.get("rebuild_count")) or 0
            merged["hit_count"] = parse_int_or_none(prior_row.get("hit_count")) or 0
            merged["exists_on_disk"] = parse_bool(prior_row.get("exists_on_disk"), default=False)

        for field in CACHE_REGISTRY_HEADERS:
            if field not in row_update or row_update[field] is None:
                continue
            value = row_update[field]
            if field in _CACHE_REGISTRY_PATH_FIELDS:
                value = clean_optional_path(value)
            elif field in _CACHE_REGISTRY_TEXT_FIELDS:
                value = clean_optional_text(value)
            merged[field] = value

        merged = self._normalize_registry_row(merged)

        if exp_id is not None:
            merged["last_exp_id"] = exp_id
        if run_id is not None:
            merged["last_run_id"] = run_id

        if error_message is not None:
            merged["last_error"] = str(error_message)

        if event_type == "cache_hit":
            merged["status"] = "ready"
            merged["hit_count"] = (parse_int_or_none(merged.get("hit_count")) or 0) + 1
            merged["last_seen_at"] = timestamp
            merged["last_validated_at"] = timestamp
            merged["last_error"] = None
        elif event_type == "build_succeeded":
            merged["status"] = "ready"
            merged["build_count"] = (parse_int_or_none(merged.get("build_count")) or 0) + 1
            if is_rebuild:
                merged["rebuild_count"] = (parse_int_or_none(merged.get("rebuild_count")) or 0) + 1
            if not clean_optional_text(merged.get("first_built_at")):
                merged["first_built_at"] = timestamp
            merged["last_built_at"] = timestamp
            merged["last_validated_at"] = timestamp
            merged["last_seen_at"] = timestamp
            merged["last_error"] = None
            if exp_id is not None and not clean_optional_text(merged.get("producer_exp_id")):
                merged["producer_exp_id"] = exp_id
            if run_id is not None and not clean_optional_text(merged.get("producer_run_id")):
                merged["producer_run_id"] = run_id
        elif event_type == "build_failed":
            merged["status"] = "failed"
            merged["last_seen_at"] = timestamp
        elif event_type == "cache_corrupted":
            merged["status"] = "corrupted"
            merged["last_seen_at"] = timestamp
            merged["last_validated_at"] = timestamp
        elif event_type == "cache_deleted":
            merged["status"] = "deleted"
            merged["exists_on_disk"] = False
            merged["file_size_bytes"] = 0
            merged["last_seen_at"] = timestamp

        return merged

    def _load_rows_locked(self) -> list[dict]:
        rows, _ = self._load_csv_rows(self.cache_registry_path)
        return [self._normalize_registry_row(row) for row in rows]

    def _load_csv_rows(self, path: str):
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return [], []

        with open(path, "r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            return list(reader), list(reader.fieldnames or [])

    def _write_csv_rows(self, path: str, headers, rows) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=headers)
            writer.writeheader()
            for row in rows:
                writer.writerow(normalize_for_json(row))
        os.replace(tmp_path, path)

    def _write_rows_locked(self, rows) -> None:
        ordered_rows = sorted(
            (self._normalize_registry_row(row) for row in rows),
            key=lambda row: (
                clean_optional_text(row.get("cache_root")) or "",
                clean_optional_text(row.get("dataset")) or "",
                clean_optional_text(row.get("task")) or "",
                clean_optional_text(row.get("artifact_type")) or "",
                clean_optional_text(row.get("relative_path")) or "",
            ),
        )
        self._write_csv_rows(self.cache_registry_path, CACHE_REGISTRY_HEADERS, ordered_rows)

    def _append_event_locked(
        self,
        *,
        row: dict,
        row_update: dict,
        prior_row: dict | None,
        event_type: str,
        exp_id,
        run_id,
        error_message: str | None,
    ) -> None:
        row_like = self._normalize_registry_row(row or row_update)
        status_before = clean_optional_text((prior_row or {}).get("status"))
        status_after = clean_optional_text(row_like.get("status")) if row_like else status_before
        event = {
            "timestamp": now_iso(),
            "event_type": event_type,
            "cache_id": clean_optional_text(row_like.get("cache_id")),
            "cache_base_id": clean_optional_text(row_like.get("cache_base_id")),
            "cache_key": clean_optional_text(row_like.get("cache_key")),
            "artifact_type": clean_optional_text(row_like.get("artifact_type")),
            "cache_root": clean_optional_path(row_like.get("cache_root")),
            "dataset": clean_optional_text(row_like.get("dataset")),
            "task": clean_optional_text(row_like.get("task")),
            "exp_id": exp_id,
            "run_id": run_id,
            "status_before": status_before,
            "status_after": status_after,
            "error_message": clean_optional_text(error_message),
            "absolute_path": clean_optional_path(row_like.get("absolute_path")),
            "relative_path": clean_optional_text(row_like.get("relative_path")),
            "config_key": clean_optional_text(row_like.get("config_key")),
            "epoch_method": clean_optional_text(row_like.get("epoch_method")),
            "pipeline": clean_optional_path(row_like.get("pipeline")),
            "pipeline_params": self._decode_json_text(row_like.get("pipeline_params")),
            "epoch_params": self._decode_json_text(row_like.get("epoch_params")),
            "file_size_bytes": parse_int_or_none(row_like.get("file_size_bytes")),
        }

        with open(self.cache_events_path, "a", encoding="utf-8") as event_file:
            event_file.write(json.dumps(normalize_for_json(event), ensure_ascii=False) + "\n")

    def _decode_json_text(self, value):
        text = clean_optional_text(value)
        if text is None:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    def _rewrite_cache_events_locked(self, cache_id_map, cache_base_map) -> None:
        if not os.path.exists(self.cache_events_path) or os.path.getsize(self.cache_events_path) == 0:
            return

        changed = False
        rewritten = []
        with open(self.cache_events_path, "r", encoding="utf-8") as event_file:
            for raw_line in event_file:
                line = raw_line.rstrip("\n")
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    rewritten.append(raw_line)
                    continue

                old_cache_id = clean_optional_text(event.get("cache_id"))
                new_cache_id = cache_id_map.get(old_cache_id, old_cache_id)
                cache_base_id = cache_base_map.get(new_cache_id) or cache_base_map.get(old_cache_id) or extract_cache_base_id(new_cache_id)

                if new_cache_id != old_cache_id:
                    event["cache_id"] = new_cache_id
                    changed = True
                if cache_base_id and clean_optional_text(event.get("cache_base_id")) != cache_base_id:
                    event["cache_base_id"] = cache_base_id
                    changed = True

                rewritten.append(json.dumps(normalize_for_json(event), ensure_ascii=False) + "\n")

        if not changed:
            return

        tmp_path = f"{self.cache_events_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as event_file:
            event_file.writelines(rewritten)
        os.replace(tmp_path, self.cache_events_path)

    def _rewrite_csv_id_columns_locked(self, path: str, columns, cache_id_map) -> None:
        rows, headers = self._load_csv_rows(path)
        if not headers:
            return

        changed = False
        for row in rows:
            for column in columns:
                if column not in row:
                    continue
                old_cache_id = clean_optional_text(row.get(column))
                new_cache_id = cache_id_map.get(old_cache_id, old_cache_id)
                if new_cache_id != old_cache_id:
                    row[column] = new_cache_id
                    changed = True

        if changed:
            self._write_csv_rows(path, headers, rows)

    def _rewrite_task_status_files_locked(self, cache_id_map) -> None:
        output_root = Path(self.output_root)
        if not output_root.exists():
            return

        for path in output_root.rglob("task_status.csv"):
            if path.parent.name != "scheduler":
                continue
            self._rewrite_csv_id_columns_locked(str(path), _CACHE_REGISTRY_ID_COLUMNS, cache_id_map)

    def _rewrite_status_json_files_locked(self, cache_id_map, cache_base_map) -> None:
        output_root = Path(self.output_root)
        if not output_root.exists():
            return

        for path in output_root.rglob("status.json"):
            if len(path.parts) < 3 or path.parts[-2] == "scheduler" or path.parts[-3] != "runs":
                continue
            self._rewrite_single_status_json_locked(str(path), cache_id_map, cache_base_map)

    def _rewrite_single_status_json_locked(self, path: str, cache_id_map, cache_base_map) -> None:
        try:
            with open(path, "r", encoding="utf-8") as status_file:
                payload = json.load(status_file)
        except (OSError, json.JSONDecodeError):
            return

        if not isinstance(payload, dict):
            return

        cache_usage = payload.get("cache_usage")
        if not isinstance(cache_usage, dict):
            return

        changed = False
        for artifact_key in ("locator", "hdf5"):
            artifact_usage = cache_usage.get(artifact_key)
            if not isinstance(artifact_usage, dict):
                continue

            old_cache_id = clean_optional_text(artifact_usage.get("cache_id"))
            new_cache_id = cache_id_map.get(old_cache_id, old_cache_id)
            cache_base_id = cache_base_map.get(new_cache_id) or cache_base_map.get(old_cache_id) or extract_cache_base_id(new_cache_id)

            if new_cache_id != old_cache_id:
                artifact_usage["cache_id"] = new_cache_id
                changed = True
            if cache_base_id and clean_optional_text(artifact_usage.get("cache_base_id")) != cache_base_id:
                artifact_usage["cache_base_id"] = cache_base_id
                changed = True

        if not changed:
            return

        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as status_file:
            json.dump(normalize_for_json(payload), status_file, ensure_ascii=False, indent=2)
            status_file.write("\n")
        os.replace(tmp_path, path)

    def _safe_sync_local_registry_locked(self, cache_root: str, rows) -> None:
        try:
            self._sync_local_registry_locked(cache_root, rows)
        except Exception as exc:
            print(f"  [cache-registry] Warning: failed to sync local registry under {cache_root}: {exc}")

    def _sync_local_registry_locked(self, cache_root: str, rows) -> None:
        subset_rows = [
            self._normalize_registry_row(row)
            for row in rows
            if clean_optional_path(row.get("cache_root")) == cache_root
        ]
        subset_rows = sorted(
            subset_rows,
            key=lambda row: (
                clean_optional_text(row.get("dataset")) or "",
                clean_optional_text(row.get("task")) or "",
                clean_optional_text(row.get("artifact_type")) or "",
                clean_optional_text(row.get("relative_path")) or "",
            ),
        )

        local_registry_dir = os.path.join(cache_root, "_registry")
        os.makedirs(local_registry_dir, exist_ok=True)
        local_registry_path = os.path.join(local_registry_dir, "cache_registry.csv")
        local_view_rows = build_cache_registry_view_rows(
            cache_rows=subset_rows,
            repo_root=self.repo_root,
        )
        self._write_csv_rows(local_registry_path, CACHE_REGISTRY_VIEW_HEADERS, local_view_rows)
