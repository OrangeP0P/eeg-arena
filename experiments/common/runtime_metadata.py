import hashlib
import json
import math
import os
from pathlib import Path


FINAL_METRIC_FIELDS = [
    "final_acc",
    "final_f1_macro",
    "final_f1_weighted",
    "final_kappa",
]


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


def normalize_subject_list(value):
    if value is None:
        return None

    if isinstance(value, str):
        text = clean_optional_text(value)
        if text is None:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = value
        else:
            if parsed is None:
                return None
            value = parsed

    if isinstance(value, (list, tuple, set)):
        normalized = []
        for item in value:
            text = clean_optional_text(item)
            if text is not None:
                normalized.append(text)
        return normalized or None

    text = clean_optional_text(value)
    if text is None:
        return None
    return [text]


def subject_list_text(value):
    return stable_json_text(normalize_subject_list(value))


def split_dataset_task(value):
    text = clean_optional_text(value)
    if text is None:
        return None, None
    if ":" not in text:
        return text, None
    dataset, task = text.split(":", 1)
    return clean_optional_text(dataset), clean_optional_text(task)


def build_data_config_id(
    *,
    dataset,
    task,
    epoch_method,
    pipeline_key,
    pipeline_params,
    epoch_params,
):
    normalized_payload = {
        "dataset": clean_optional_text(dataset),
        "task": clean_optional_text(task),
        "epoch_method": clean_optional_text(epoch_method),
        "pipeline_key": clean_optional_text(pipeline_key),
        "pipeline_params": normalize_for_json(pipeline_params),
        "epoch_params": normalize_for_json(epoch_params),
    }
    canonical = stable_json_text(normalized_payload) or ""
    return f"dcfg_{hashlib.sha1(canonical.encode('utf-8')).hexdigest()[:12]}"


def resolve_data_config_fields(
    *,
    dataset,
    task,
    epoch_method,
    pipeline_key,
    pipeline_params,
    epoch_params,
    data_config_key,
    data_config_id=None,
):
    normalized_key = clean_optional_text(data_config_key)
    normalized_id = clean_optional_text(data_config_id)
    if normalized_id is None:
        required_values = (
            clean_optional_text(dataset),
            clean_optional_text(task),
            clean_optional_text(epoch_method),
            clean_optional_text(pipeline_key),
        )
        if all(required_values):
            normalized_id = build_data_config_id(
                dataset=dataset,
                task=task,
                epoch_method=epoch_method,
                pipeline_key=pipeline_key,
                pipeline_params=pipeline_params,
                epoch_params=epoch_params,
            )
    return {
        "data_config_id": normalized_id,
        "data_config_key": normalized_key,
    }


def extract_final_metric_fields(final_test_payload):
    metrics = {}
    if isinstance(final_test_payload, dict):
        maybe_metrics = final_test_payload.get("metrics")
        if isinstance(maybe_metrics, dict):
            metrics = maybe_metrics

    return {
        "final_acc": metrics.get("acc"),
        "final_f1_macro": metrics.get("f1_macro"),
        "final_f1_weighted": metrics.get("f1_weighted"),
        "final_kappa": metrics.get("kappa"),
    }
