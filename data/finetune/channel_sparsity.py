from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


DEFAULT_CHANNEL_SPARSITY_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "configs"
    / "low"
    / "table_v"
    / "table_v_channel_config.json"
)


def _dataset_task_key(dataset_name: str, task_name: str) -> str:
    return f"{dataset_name}:{task_name}"


def _canonical_channel_token(name: str) -> str:
    normalized = str(name).strip().replace(" ", "")
    if ":" in normalized:
        prefix, rest = normalized.split(":", 1)
        return f"{prefix.lower()}:{rest.upper()}"
    return normalized.upper()


def _equivalent_tokens(name: str) -> tuple[str, ...]:
    token = _canonical_channel_token(name)
    if ":" in token:
        return (token,)
    return (token, f"eeg:{token}")


def _resolve_config_path(config_path=None) -> str:
    if config_path is None:
        return str(DEFAULT_CHANNEL_SPARSITY_CONFIG)
    return str(Path(config_path).expanduser().resolve())


@lru_cache(maxsize=16)
def _load_channel_sparsity_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Channel sparsity config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict) or not isinstance(payload.get("datasets"), dict):
        raise ValueError(f"Invalid channel sparsity config '{path}': missing object field 'datasets'.")
    return payload


def _variant_channels_from_config(payload: dict, dataset_name: str, task_name: str, channel_variant: str) -> list[str]:
    key = _dataset_task_key(dataset_name, task_name)
    dataset_entry = payload["datasets"].get(key)
    if not isinstance(dataset_entry, dict):
        available = ", ".join(sorted(payload["datasets"].keys()))
        raise KeyError(f"Dataset task '{key}' is not present in channel sparsity config. Available: {available}")

    variants = dataset_entry.get("variants")
    if not isinstance(variants, dict):
        raise ValueError(f"Invalid channel sparsity config entry for '{key}': missing object field 'variants'.")

    variant_entry = variants.get(channel_variant)
    if variant_entry is None:
        available = ", ".join(sorted(variants.keys()))
        raise KeyError(f"Unknown channel sparsity variant '{channel_variant}' for '{key}'. Available: {available}")

    channels = variant_entry.get("channels") if isinstance(variant_entry, dict) else variant_entry
    if not isinstance(channels, list) or not channels or not all(isinstance(channel, str) for channel in channels):
        raise ValueError(
            f"Invalid channel sparsity variant '{channel_variant}' for '{key}': "
            "expected a non-empty channel-name list."
        )
    if isinstance(variant_entry, dict) and variant_entry.get("target_count") is not None:
        expected = int(variant_entry["target_count"])
        if expected != len(channels):
            raise ValueError(
                f"Invalid channel sparsity variant '{channel_variant}' for '{key}': "
                f"target_count={expected}, but channels has length {len(channels)}."
            )
    return list(channels)


def _validate_channels_available(dataset_name: str, task_name: str, channel_variant: str, channels: list[str], available_channels):
    if available_channels is None:
        return

    available_tokens = set()
    for channel in available_channels:
        available_tokens.update(_equivalent_tokens(channel))

    missing = []
    for channel in channels:
        if not any(token in available_tokens for token in _equivalent_tokens(channel)):
            missing.append(channel)

    if missing:
        key = _dataset_task_key(dataset_name, task_name)
        raise ValueError(
            f"Channel sparsity variant '{channel_variant}' for '{key}' contains channels "
            f"that are absent from this dataset cache: {missing}"
        )


def resolve_channel_sparsity_variant(
    dataset_name: str,
    task_name: str,
    channel_variant: str,
    available_channels=None,
    *,
    config_path=None,
) -> dict:
    config_path = _resolve_config_path(config_path)
    payload = _load_channel_sparsity_config(config_path)
    channels = _variant_channels_from_config(payload, dataset_name, task_name, channel_variant)
    _validate_channels_available(dataset_name, task_name, channel_variant, channels, available_channels)
    return {
        "name": channel_variant,
        "target_count": len(channels),
        "selected_channels": channels,
        "config_path": config_path,
    }
