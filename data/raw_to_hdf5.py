"""
raw_to_hdf5.py — Dynamic HDF5 builder for EEG-Arena task-based datasets.

Workflow:
1. Check <cache_dir>/<dataset_name>/<task_name>/<config_key>.hdf5
2. If missing: check <cache_dir>/<dataset_name>/locator.csv
3. If locator missing: scan raw_path with eegunity to build locator, then cache it
4. Run epoch_by_*_hdf5 to create HDF5 file

The config_key encodes: epoch_method_prefix + task_name + epoch_params (sorted) + pipeline_key
This ensures different preprocessing configurations produce different cached files.
"""
import fcntl
import hashlib
import inspect
import os
import sys
import importlib.util
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, List, Tuple

from experiments.common.cache_registry import (
    CacheRegistry,
    build_cache_base_id,
    build_hdf5_cache_id,
    build_hdf5_cache_key,
    build_locator_cache_id,
    build_locator_cache_key,
    cache_relative_path,
    read_hdf5_stats,
    read_locator_stats,
    stable_json_text,
)
from experiments.common.runtime_metadata import resolve_data_config_fields

# Ensure project root is on path for eegunity import
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_CACHE_REGISTRY = CacheRegistry(_PROJECT_ROOT)

# Valid epoch param keys per epoch method
_VALID_EPOCH_PARAMS = {
    'epoch_by_event_hdf5':        {'tmin', 'tmax'},
    'epoch_by_segmentation_hdf5': {'segment_length', 'overlap'},
    'epoch_by_long_event_hdf5':   {'tmin', 'tmax', 'overlap'},
}

_CHANNEL_POLICY_NAME = "max_common_eeg_v1"


@contextmanager
def _exclusive_file_lock(lock_path: str, label: str):
    """
    Acquire an exclusive advisory lock for a cache artifact.

    The first process proceeds immediately. Contending processes block until the
    active build finishes, then re-check the cache on disk.

    Uses fcntl.lockf() (POSIX record locks) rather than flock() because flock()
    is process-local and does not propagate across NFS/GPFS nodes; lockf() is
    honoured by the NFS lock manager and works cross-node on shared HPC filesystems.
    """
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    with open(lock_path, 'a+', encoding='utf-8') as lock_file:
        waited = False
        try:
            fcntl.lockf(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError):
            waited = True
            print(f"  [{label}] Waiting for concurrent cache build: {lock_path}")
            fcntl.lockf(lock_file, fcntl.LOCK_EX)
        try:
            if waited:
                print(f"  [{label}] Build lock acquired, re-checking cache state.")
            yield
        finally:
            fcntl.lockf(lock_file, fcntl.LOCK_UN)


@dataclass
class HDF5BuildParams:
    """Parameters that control HDF5 building and cache key generation.

    epoch_params may be None or a partial dict — ensure_hdf5() fills in
    missing keys from task_config['default_epoch_params'].

    Resampling and other preprocessing is handled entirely inside the pipeline.
    """
    epoch_params: Optional[Dict] = None   # e.g. {'tmin': -0.5, 'tmax': 4.0}
    pipeline: Optional[str] = None        # path to pipeline .py file
    pipeline_params: Optional[Dict] = None  # kwargs forwarded to make_pipeline()
    rebuild_hdf5: bool = False
    num_workers: int = 16
    exp_id: Optional[int] = None
    run_id: Optional[str] = None
    usage_report: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pipeline loading
# ---------------------------------------------------------------------------

def load_pipeline(
    pipeline_path: Optional[str],
    pipeline_params: Optional[Dict] = None,
) -> Tuple[Optional[Callable], str, Optional[str]]:
    """
    Dynamically load a pipeline .py file and return a ready-to-call pipeline.

    Two pipeline interfaces are supported:

    1. **Parameterised** (``make_pipeline`` factory):
       The file exports ``make_pipeline(**kwargs) -> Callable[[raw], raw]``.
       ``pipeline_params`` is forwarded as keyword arguments.
       Every key in ``pipeline_params`` must match a parameter of ``make_pipeline``
       — extra or unknown keys raise ``ValueError``.

    2. **No-param** (plain ``pipeline`` function):
       The file exports ``def pipeline(raw) -> raw``.
       ``pipeline_params`` must be ``None`` or empty.

    Cache key (returned as second element):
        ``<stem>_<k1><v1>_<k2><v2>...`` (params sorted by key), or just ``<stem>``
        when no params are given, or ``'nopipe'`` when ``pipeline_path`` is ``None``.

    Args:
        pipeline_path: Path to the pipeline .py file (absolute or relative to
                       project root). ``None`` means no pipeline.
        pipeline_params: Dict of parameters to forward to ``make_pipeline``.

    Returns:
        (pipeline_callable, pipeline_key)
    """
    if pipeline_path is None:
        if pipeline_params:
            raise ValueError(
                "--pipeline_param was given but --pipeline was not specified."
            )
        return None, 'nopipe', None

    # Resolve relative path from the project root
    if not os.path.isabs(pipeline_path):
        pipeline_path = os.path.join(_PROJECT_ROOT, pipeline_path)

    if not os.path.isfile(pipeline_path):
        raise FileNotFoundError(f"Pipeline file not found: {pipeline_path}")

    stem = os.path.splitext(os.path.basename(pipeline_path))[0]

    # Load the module dynamically
    spec = importlib.util.spec_from_file_location("_user_pipeline", pipeline_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    has_factory = hasattr(mod, 'make_pipeline') and callable(mod.make_pipeline)
    has_plain   = hasattr(mod, 'pipeline')      and callable(mod.pipeline)

    if pipeline_params:
        # Requires make_pipeline factory
        if not has_factory:
            raise AttributeError(
                f"--pipeline_param was given but '{pipeline_path}' does not export "
                f"a callable 'make_pipeline'. Add a make_pipeline() factory or "
                f"remove --pipeline_param."
            )
        # Validate that every supplied key is a valid parameter
        valid_keys = set(inspect.signature(mod.make_pipeline).parameters.keys())
        unknown = set(pipeline_params.keys()) - valid_keys
        if unknown:
            raise ValueError(
                f"--pipeline_param contains unknown key(s) {sorted(unknown)} for "
                f"make_pipeline() in '{pipeline_path}'. "
                f"Valid keys: {sorted(valid_keys)}"
            )
        pipeline_fn = mod.make_pipeline(**pipeline_params)
        param_str = '_'.join(f"{k}{v}" for k, v in sorted(pipeline_params.items()))
        pipeline_key = f"{stem}_{param_str}"
    else:
        # No params supplied
        if has_plain:
            pipeline_fn = mod.pipeline
        elif has_factory:
            # make_pipeline with all-default params
            sig = inspect.signature(mod.make_pipeline)
            missing = [
                p.name for p in sig.parameters.values()
                if p.default is inspect.Parameter.empty
            ]
            if missing:
                raise ValueError(
                    f"make_pipeline() in '{pipeline_path}' has required parameter(s) "
                    f"{missing} but --pipeline_param was not given."
                )
            pipeline_fn = mod.make_pipeline()
        else:
            raise AttributeError(
                f"Pipeline file '{pipeline_path}' must export either a callable "
                f"'make_pipeline' (factory) or a callable 'pipeline'."
            )
        pipeline_key = stem

    if not callable(pipeline_fn):
        raise TypeError(
            f"Pipeline returned by '{pipeline_path}' is not callable, "
            f"got {type(pipeline_fn)}"
        )

    return pipeline_fn, pipeline_key, pipeline_path


def _update_usage_report(usage_report: Optional[Dict], payload: Dict) -> None:
    if usage_report is None:
        return
    usage_report.clear()
    usage_report.update(payload)


def _extract_typed_eeg_channels(channel_string: str) -> List[str]:
    """Return the EEG channels from one locator row in stable order."""
    from eegunity.modules.parser.eeg_parser import channel_name_parser

    formatted = channel_name_parser(str(channel_string))
    eeg_channels: List[str] = []
    seen = set()
    for entry in formatted.split(','):
        entry = entry.strip()
        if not entry or ':' not in entry:
            continue
        prefix, name = entry.split(':', 1)
        if prefix.strip().lower() != 'eeg':
            continue
        name = name.strip()
        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)
        eeg_channels.append(name)
    return eeg_channels


def resolve_common_eeg_channels(locator_path: str, declared_channels: Optional[List[str]] = None) -> Dict:
    """Compute the EEG channel layout for one task.

    When ``declared_channels`` is provided, that registry-declared layout is
    used directly. This is required for some ``HANDLES_UNAVAILABLE`` kernels
    whose raw files cannot yield reliable channel metadata during locator scan.
    """
    if declared_channels:
        ordered_channels = [str(channel) for channel in declared_channels]
        signature_hash = hashlib.sha1(
            ','.join(ordered_channels).encode('utf-8')
        ).hexdigest()[:8]
        signature = f"{len(ordered_channels)}-{signature_hash}"
        return {
            'policy': 'registry_declared_eeg_v1',
            'channels': ordered_channels,
            'count': len(ordered_channels),
            'signature': signature,
            'files_considered': 0,
        }

    import pandas as pd

    locator = pd.read_csv(locator_path)
    if 'Completeness Check' not in locator.columns:
        raise ValueError(
            f"Locator is missing the 'Completeness Check' column: {locator_path}"
        )

    available = locator[locator['Completeness Check'].astype(str) != 'Unavailable']
    if available.empty:
        raise ValueError(
            f"No available recordings were found in locator: {locator_path}"
        )

    common_keys = None
    ordered_keys = None
    canonical_by_key: Dict[str, str] = {}
    files_considered = 0

    for _, row in available.iterrows():
        eeg_channels = _extract_typed_eeg_channels(row.get('Channel Names', ''))
        file_path = str(row.get('File Path', 'unknown'))
        if not eeg_channels:
            raise ValueError(
                "Unable to derive a common EEG channel set because at least one "
                f"recording has no EEG channels after locator formatting: {file_path}"
            )

        row_keys = [channel.casefold() for channel in eeg_channels]
        row_key_set = set(row_keys)
        for channel, key in zip(eeg_channels, row_keys):
            canonical_by_key.setdefault(key, channel)

        if ordered_keys is None:
            ordered_keys = row_keys
            common_keys = set(row_key_set)
        else:
            common_keys &= row_key_set
        files_considered += 1

    ordered_channels = [
        canonical_by_key[key]
        for key in (ordered_keys or [])
        if key in (common_keys or set())
    ]
    if not ordered_channels:
        raise ValueError(
            "No common EEG channels were found across the available recordings in "
            f"{locator_path}. The current kernel/dataset combination cannot produce "
            "a single flat HDF5 with a uniform channel layout."
        )

    signature_hash = hashlib.sha1(
        ','.join(ordered_channels).encode('utf-8')
    ).hexdigest()[:8]
    signature = f"{len(ordered_channels)}-{signature_hash}"
    return {
        'policy': _CHANNEL_POLICY_NAME,
        'channels': ordered_channels,
        'count': len(ordered_channels),
        'signature': signature,
        'files_considered': files_considered,
    }


def _task_recording_filter_token(task_config: Dict) -> Optional[str]:
    recording_filter = task_config.get('recording_filter')
    if not recording_filter:
        return None
    filter_text = stable_json_text(recording_filter) or ''
    return hashlib.sha1(filter_text.encode('utf-8')).hexdigest()[:8]


def _apply_task_recording_filter(
    *,
    source_locator_info: Dict,
    task_config: Dict,
    cache_root: str,
    dataset_name: str,
    task_name: str,
) -> Dict:
    """Return a task-specific locator copy when a recording filter is configured."""
    recording_filter = task_config.get('recording_filter')
    if not recording_filter:
        return source_locator_info

    basename_regex = recording_filter.get('basename_regex')
    required_eeg_channels = recording_filter.get('required_eeg_channels')
    if basename_regex is None and required_eeg_channels is None:
        raise ValueError(
            f"recording_filter for '{dataset_name}:{task_name}' must define "
            "at least one of 'basename_regex' or 'required_eeg_channels'."
        )

    import pandas as pd
    import re

    pattern = None
    if basename_regex is not None:
        try:
            pattern = re.compile(basename_regex)
        except re.error as exc:
            raise ValueError(
                f"Invalid basename_regex for '{dataset_name}:{task_name}': {exc}"
            ) from exc

    required_eeg_channel_keys = None
    if required_eeg_channels is not None:
        if not isinstance(required_eeg_channels, list) or not required_eeg_channels:
            raise ValueError(
                f"required_eeg_channels for '{dataset_name}:{task_name}' must be "
                "a non-empty list of strings."
            )
        required_eeg_channel_keys = []
        for channel in required_eeg_channels:
            if not isinstance(channel, str) or not channel.strip():
                raise ValueError(
                    f"required_eeg_channels for '{dataset_name}:{task_name}' must "
                    "contain only non-empty strings."
                )
            required_eeg_channel_keys.append(channel.strip().casefold())

    source_path = source_locator_info['path']
    locator = pd.read_csv(source_path)
    if 'File Path' not in locator.columns:
        raise ValueError(f"Locator is missing the 'File Path' column: {source_path}")

    def _matches_row(row) -> bool:
        if pattern is not None:
            path = str(row.get('File Path', ''))
            if not pattern.search(os.path.basename(path)):
                return False

        if required_eeg_channel_keys is not None:
            row_eeg_channels = _extract_typed_eeg_channels(row.get('Channel Names', ''))
            row_eeg_keys = {channel.casefold() for channel in row_eeg_channels}
            if not all(channel in row_eeg_keys for channel in required_eeg_channel_keys):
                return False

        return True

    mask = locator.apply(_matches_row, axis=1)
    filtered = locator[mask].copy()
    if filtered.empty:
        criteria = []
        if basename_regex is not None:
            criteria.append(f"basename_regex={basename_regex!r}")
        if required_eeg_channels is not None:
            criteria.append(f"required_eeg_channels={required_eeg_channels!r}")
        raise ValueError(
            f"recording_filter for '{dataset_name}:{task_name}' matched zero "
            f"recordings: {', '.join(criteria)}"
        )

    filtered_path = os.path.abspath(
        os.path.join(cache_root, dataset_name, task_name, 'locator.csv')
    )
    os.makedirs(os.path.dirname(filtered_path), exist_ok=True)
    tmp_path = f"{filtered_path}.tmp"
    filtered.to_csv(tmp_path, index=False)
    os.replace(tmp_path, filtered_path)

    filtered_info = dict(source_locator_info)
    filtered_info['absolute_path'] = filtered_path
    filtered_info['path'] = filtered_path
    filtered_info['relative_path'] = cache_relative_path(cache_root, filtered_path)
    filtered_info['recording_filter'] = stable_json_text(recording_filter)
    return filtered_info


def _write_hdf5_channel_metadata(hdf5_path: str, common_channel_info: Dict) -> None:
    """Persist enforced channel-layout metadata into the HDF5 root attrs."""
    import h5py

    with h5py.File(hdf5_path, 'a') as handle:
        handle.attrs['channel_policy'] = str(common_channel_info['policy'])
        handle.attrs['common_eeg_channels'] = stable_json_text(common_channel_info['channels'])
        handle.attrs['common_eeg_channel_count'] = int(common_channel_info['count'])
        handle.attrs['common_eeg_channel_signature'] = str(common_channel_info['signature'])


def _validate_locator_cache(locator_path: str) -> Tuple[bool, Optional[str]]:
    """
    Lightweight integrity check for cached locator CSVs.

    We only need to detect obviously incomplete files left behind by interrupted
    concurrent writes, so a header-level check is sufficient here.
    """
    try:
        if not os.path.exists(locator_path):
            return False, "missing"
        if os.path.getsize(locator_path) == 0:
            return False, "empty file"
        with open(locator_path, 'r', encoding='utf-8') as handle:
            header = handle.readline().strip()
        if not header:
            return False, "missing CSV header"
        if 'File Path' not in header:
            return False, f"unexpected CSV header: {header}"
        return True, None
    except Exception as exc:
        return False, str(exc)


def _validate_hdf5_cache(hdf5_path: str) -> Tuple[bool, Optional[str]]:
    """
    Validate that a cached HDF5 file is readable enough for downstream use.

    Previous interrupted concurrent builds can leave behind files whose metadata
    exists but whose compressed dataset chunks cannot be read. We detect that by
    opening the file and touching representative samples.
    """
    try:
        import h5py

        if not os.path.exists(hdf5_path):
            return False, "missing"
        if os.path.getsize(hdf5_path) == 0:
            return False, "empty file"

        with h5py.File(hdf5_path, 'r') as handle:
            if 'data' not in handle:
                return False, "missing 'data' dataset"
            data_ds = handle['data']
            if len(data_ds.shape) != 3:
                return False, f"unexpected data shape: {data_ds.shape}"
            n_epochs = data_ds.shape[0]
            if n_epochs > 0:
                sample_indices = [0]
                if n_epochs > 1:
                    sample_indices.append(n_epochs - 1)
                for idx in sample_indices:
                    _ = data_ds[idx]
        return True, None
    except Exception as exc:
        return False, str(exc)


def _remove_if_exists(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)


def _safe_record_cache_event(**kwargs):
    try:
        return _CACHE_REGISTRY.record_event(**kwargs)
    except Exception as exc:
        print(f"  [cache-registry] Warning: {exc}")
        return None


def _normalize_kernel_spec(kernel_spec: Optional[str]) -> Optional[str]:
    if kernel_spec is None:
        return None
    return os.path.abspath(kernel_spec)


def _build_locator_cache_info(
    *,
    cache_root: str,
    dataset_name: str,
    raw_path: str,
    domain_tag: str,
    kernel_spec: Optional[str],
) -> Dict:
    absolute_path = os.path.abspath(os.path.join(cache_root, dataset_name, 'locator.csv'))
    relative_path = cache_relative_path(cache_root, absolute_path)
    cache_base_id = build_cache_base_id(
        cache_root=cache_root,
        dataset=dataset_name,
        raw_path=raw_path,
        domain_tag=domain_tag,
        kernel_spec=kernel_spec,
    )
    return {
        'artifact_type': 'locator',
        'cache_root': os.path.abspath(cache_root),
        'dataset': dataset_name,
        'task': None,
        'cache_base_id': cache_base_id,
        'parent_locator_cache_id': None,
        'raw_path': os.path.abspath(raw_path),
        'domain_tag': domain_tag,
        'kernel_spec': _normalize_kernel_spec(kernel_spec),
        'relative_path': relative_path,
        'absolute_path': absolute_path,
        'cache_id': build_locator_cache_id(cache_base_id),
        'cache_key': build_locator_cache_key(dataset_name, domain_tag, kernel_spec),
    }


def _build_hdf5_cache_info(
    *,
    cache_root: str,
    dataset_name: str,
    task_name: str,
    raw_path: str,
    domain_tag: str,
    kernel_spec: Optional[str],
    config_key: str,
    epoch_method: str,
    pipeline_path: Optional[str],
    pipeline_key: str,
    pipeline_params: Optional[Dict],
    epoch_params: Dict,
    parent_locator_cache_id: Optional[str],
    channel_policy: Optional[str] = None,
    common_eeg_channels: Optional[List[str]] = None,
    common_eeg_signature: Optional[str] = None,
) -> Dict:
    absolute_path = os.path.abspath(os.path.join(cache_root, dataset_name, task_name, f"{config_key}.hdf5"))
    relative_path = cache_relative_path(cache_root, absolute_path)
    cache_base_id = build_cache_base_id(
        cache_root=cache_root,
        dataset=dataset_name,
        raw_path=raw_path,
        domain_tag=domain_tag,
        kernel_spec=kernel_spec,
    )
    return {
        'artifact_type': 'hdf5',
        'cache_root': os.path.abspath(cache_root),
        'dataset': dataset_name,
        'task': task_name,
        'cache_base_id': cache_base_id,
        'parent_locator_cache_id': parent_locator_cache_id or build_locator_cache_id(cache_base_id),
        'raw_path': os.path.abspath(raw_path),
        'domain_tag': domain_tag,
        'kernel_spec': _normalize_kernel_spec(kernel_spec),
        'relative_path': relative_path,
        'absolute_path': absolute_path,
        'epoch_method': epoch_method,
        'config_key': config_key,
        'pipeline': os.path.abspath(pipeline_path) if pipeline_path else None,
        'pipeline_key': pipeline_key,
        'pipeline_params': stable_json_text(pipeline_params),
        'epoch_params': stable_json_text(epoch_params),
        'channel_policy': channel_policy,
        'common_eeg_channels': stable_json_text(common_eeg_channels),
        'common_eeg_channel_count': len(common_eeg_channels or []),
        'common_eeg_signature': common_eeg_signature,
        'cache_id': build_hdf5_cache_id(cache_base_id, task_name, config_key),
        'cache_key': build_hdf5_cache_key(dataset_name, task_name, config_key),
    }


def _apply_file_state(row_update: Dict, path: str) -> Dict:
    enriched = dict(row_update)
    exists = os.path.exists(path)
    enriched['exists_on_disk'] = exists
    enriched['file_size_bytes'] = os.path.getsize(path) if exists else 0
    return enriched


def _augment_locator_stats(row_update: Dict, locator_path: str) -> Dict:
    enriched = _apply_file_state(row_update, locator_path)
    enriched.update(read_locator_stats(locator_path))
    return enriched


def _augment_hdf5_stats(row_update: Dict, hdf5_path: str) -> Dict:
    enriched = _apply_file_state(row_update, hdf5_path)
    try:
        enriched.update(read_hdf5_stats(hdf5_path))
    except Exception as exc:
        enriched.setdefault('last_error', str(exc))
    return enriched


def _build_cache_usage(
    *,
    cache_root: str,
    dataset_name: str,
    task_name: str,
    epoch_method: str,
    pipeline_path: Optional[str],
    pipeline_key: str,
    pipeline_params: Optional[Dict],
    epoch_params: Dict,
    locator_info: Dict,
    hdf5_info: Dict,
    common_channel_info: Dict,
) -> Dict:
    return {
        **resolve_data_config_fields(
            dataset=dataset_name,
            task=task_name,
            epoch_method=epoch_method,
            pipeline_key=pipeline_key,
            pipeline_params=pipeline_params,
            epoch_params=epoch_params,
            data_config_key=hdf5_info.get('config_key'),
            data_config_id=hdf5_info.get('data_config_id'),
        ),
        'cache_root': os.path.abspath(cache_root),
        'dataset': dataset_name,
        'task': task_name,
        'epoch_method': epoch_method,
        'pipeline': os.path.abspath(pipeline_path) if pipeline_path else None,
        'pipeline_key': pipeline_key,
        'pipeline_params': dict(pipeline_params) if pipeline_params else None,
        'epoch_params': dict(epoch_params),
        'channel_policy': common_channel_info.get('policy'),
        'common_eeg_channels': list(common_channel_info.get('channels') or []),
        'common_eeg_channel_count': int(common_channel_info.get('count') or 0),
        'common_eeg_channel_signature': common_channel_info.get('signature'),
        'locator': {
            'cache_id': locator_info.get('cache_id'),
            'cache_base_id': locator_info.get('cache_base_id'),
            'cache_key': locator_info.get('cache_key'),
            'absolute_path': locator_info.get('absolute_path'),
            'relative_path': locator_info.get('relative_path'),
        },
        'hdf5': {
            'cache_id': hdf5_info.get('cache_id'),
            'cache_base_id': hdf5_info.get('cache_base_id'),
            'cache_key': hdf5_info.get('cache_key'),
            'config_key': hdf5_info.get('config_key'),
            'absolute_path': hdf5_info.get('absolute_path'),
            'relative_path': hdf5_info.get('relative_path'),
        },
    }


# ---------------------------------------------------------------------------
# HDF5 filename computation
# ---------------------------------------------------------------------------

def compute_config_key(
    epoch_method: str,
    task_name: str,
    resolved_epoch_params: Dict,
    pipeline_key: str,
    channel_layout_token: Optional[str] = None,
) -> str:
    """
    Compute a human-readable config key that uniquely identifies a preprocessing config.

    Format:
        event_{task_name}_{k1v1_sorted}_{pipeline_key}_{channel_layout_token}
        seg_{task_name}_{k1v1_sorted}_{pipeline_key}_{channel_layout_token}
        longevent_{task_name}_{k1v1_sorted}_{pipeline_key}_{channel_layout_token}

    Args:
        epoch_method: One of 'epoch_by_event_hdf5', 'epoch_by_segmentation_hdf5',
                      'epoch_by_long_event_hdf5'
        task_name: Task name (used in cache key for human readability)
        resolved_epoch_params: Fully resolved epoch params dict (no None values)
        pipeline_key: Human-readable pipeline identifier returned by load_pipeline().
        channel_layout_token: Optional token describing the enforced channel layout.

    Returns:
        Config key string (used as HDF5 file_name_prefix)
    """
    valid_keys = _VALID_EPOCH_PARAMS.get(epoch_method)
    if valid_keys is None:
        raise ValueError(
            f"Unknown epoch_method: '{epoch_method}'. "
            f"Expected one of: {list(_VALID_EPOCH_PARAMS.keys())}"
        )

    unknown = set(resolved_epoch_params.keys()) - valid_keys
    if unknown:
        raise ValueError(
            f"epoch_params contains invalid key(s) {sorted(unknown)} for "
            f"epoch_method='{epoch_method}'. Valid keys: {sorted(valid_keys)}"
        )

    if epoch_method == 'epoch_by_event_hdf5':
        prefix = 'event'
    elif epoch_method == 'epoch_by_segmentation_hdf5':
        prefix = 'seg'
    else:
        prefix = 'longevent'

    params_str = '_'.join(
        f"{k}{v}" for k, v in sorted(resolved_epoch_params.items())
    )
    parts = [prefix, task_name, params_str, pipeline_key]
    if channel_layout_token:
        parts.append(channel_layout_token)
    return '_'.join(parts)


# ---------------------------------------------------------------------------
# Locator cache
# ---------------------------------------------------------------------------

def ensure_locator(
    dataset_name: str,
    raw_path: str,
    domain_tag: str,
    kernel_spec: Optional[str],
    num_workers: int,
    locator_cache_dir: str,
    exp_id: Optional[int] = None,
    run_id: Optional[str] = None,
) -> Dict:
    """
    Ensure a locator CSV exists for the given raw dataset.

    Cache location: <locator_cache_dir>/locator.csv
    The directory is created automatically if it does not exist.

    Args:
        raw_path: Path to the raw EEG dataset directory.
        domain_tag: Domain tag for UnifiedDataset.
        kernel_spec: Path to kernel .py file, or None for generic BIDS scanning.
        num_workers: Number of parallel workers for scanning.
        locator_cache_dir: Directory to store the locator CSV (dataset-level, shared).

    Returns:
        Locator cache metadata, including ``path`` / ``cache_id`` / ``cache_key``.
    """
    cached_locator = os.path.join(locator_cache_dir, 'locator.csv')
    lock_path = os.path.join(locator_cache_dir, '.locator.lock')
    cache_root = os.path.abspath(os.path.dirname(locator_cache_dir))
    locator_info = _build_locator_cache_info(
        cache_root=cache_root,
        dataset_name=dataset_name,
        raw_path=raw_path,
        domain_tag=domain_tag,
        kernel_spec=kernel_spec,
    )
    locator_info['path'] = locator_info['absolute_path']

    if os.path.exists(cached_locator):
        is_valid, reason = _validate_locator_cache(cached_locator)
        if is_valid:
            print(f"  [locator] Cache hit: {cached_locator}")
            _safe_record_cache_event(
                row_update=_augment_locator_stats(locator_info, cached_locator),
                event_type="cache_hit",
                exp_id=exp_id,
                run_id=run_id,
            )
            return locator_info
        print(f"  [locator] Corrupted cache detected ({reason}) — rebuilding: {cached_locator}")
        _safe_record_cache_event(
            row_update=_augment_locator_stats(locator_info, cached_locator),
            event_type="cache_corrupted",
            exp_id=exp_id,
            run_id=run_id,
            error_message=reason,
        )
        _remove_if_exists(cached_locator)
        _safe_record_cache_event(
            row_update=_apply_file_state(locator_info, cached_locator),
            event_type="cache_deleted",
            exp_id=exp_id,
            run_id=run_id,
            error_message="removed corrupted locator cache before rebuild",
        )

    os.makedirs(locator_cache_dir, exist_ok=True)
    with _exclusive_file_lock(lock_path, 'locator'):
        if os.path.exists(cached_locator):
            is_valid, reason = _validate_locator_cache(cached_locator)
            if is_valid:
                print(f"  [locator] Cache hit after wait: {cached_locator}")
                _safe_record_cache_event(
                    row_update=_augment_locator_stats(locator_info, cached_locator),
                    event_type="cache_hit",
                    exp_id=exp_id,
                    run_id=run_id,
                )
                return locator_info
            print(f"  [locator] Corrupted cache detected after wait ({reason}) — rebuilding: {cached_locator}")
            _safe_record_cache_event(
                row_update=_augment_locator_stats(locator_info, cached_locator),
                event_type="cache_corrupted",
                exp_id=exp_id,
                run_id=run_id,
                error_message=reason,
            )
            _remove_if_exists(cached_locator)
            _safe_record_cache_event(
                row_update=_apply_file_state(locator_info, cached_locator),
                event_type="cache_deleted",
                exp_id=exp_id,
                run_id=run_id,
                error_message="removed corrupted locator cache before rebuild",
            )

        print(f"  [locator] Cache miss — scanning raw dataset: {raw_path}")
        _safe_record_cache_event(
            row_update=_apply_file_state(locator_info, cached_locator),
            event_type="build_started",
            exp_id=exp_id,
            run_id=run_id,
            update_registry=False,
        )

        try:
            from eegunity import UnifiedDataset
            u_ds = UnifiedDataset(
                dataset_path=raw_path,
                domain_tag=domain_tag,
                kernel_spec=kernel_spec,
                num_workers=num_workers,
                verbose='WARNING',
            )
            u_ds.eeg_batch.format_channel_names()
            u_ds.eeg_batch.auto_domain()

            u_ds.save_locator(cached_locator)
            print(f"  [locator] Saved to: {cached_locator}")
            _safe_record_cache_event(
                row_update=_augment_locator_stats(locator_info, cached_locator),
                event_type="build_succeeded",
                exp_id=exp_id,
                run_id=run_id,
            )
            return locator_info
        except Exception as exc:
            _safe_record_cache_event(
                row_update=_apply_file_state(locator_info, cached_locator),
                event_type="build_failed",
                exp_id=exp_id,
                run_id=run_id,
                error_message=str(exc),
            )
            raise


# ---------------------------------------------------------------------------
# HDF5 builder
# ---------------------------------------------------------------------------

def ensure_hdf5(
    dataset_name: str,
    task_name: str,
    dataset_config: dict,
    task_config: dict,
    build_params: HDF5BuildParams,
    cache_dir: str,
) -> str:
    """
    Ensure a processed HDF5 file exists for the given dataset + task + build parameters.

    Cache layout:
        <cache_dir>/<dataset_name>/locator.csv          (shared across tasks)
        <cache_dir>/<dataset_name>/<task_name>/<config_key>.hdf5

    Cache lookup:
    1. Check <cache_dir>/<dataset_name>/<task_name>/<config_key>.hdf5
    2. If exists and rebuild_hdf5=False → return immediately
    3. Otherwise → build via ensure_locator + eegunity epoch functions

    Args:
        dataset_name: Registered dataset name (lowercase).
        task_name: Registered task name within the dataset (lowercase).
        dataset_config: Dataset-level config dict (raw_path, domain_tag, kernel_spec).
        task_config: Task-level config dict (epoch_method, event_dict, default_epoch_params).
        build_params: HDF5BuildParams controlling preprocessing and cache key.
        cache_dir: Base cache directory (e.g. from --cache_dir CLI param).

    Returns:
        Absolute path to the HDF5 file.
    """
    epoch_method = task_config['epoch_method']

    # Resolve epoch_params: CLI overrides > task defaults
    defaults = task_config.get('default_epoch_params', {})
    valid_keys = _VALID_EPOCH_PARAMS[epoch_method]
    resolved_epoch_params = {}
    for key in valid_keys:
        cli_val = (build_params.epoch_params or {}).get(key)
        if cli_val is not None:
            resolved_epoch_params[key] = cli_val
        elif key in defaults:
            resolved_epoch_params[key] = defaults[key]
        else:
            raise ValueError(
                f"Epoch param '{key}' is required for {epoch_method} but was not "
                f"provided via --epoch_param and has no default in task_config "
                f"'{dataset_name}:{task_name}'."
            )

    # Validate any extra keys the user passed
    if build_params.epoch_params:
        unknown = set(build_params.epoch_params.keys()) - valid_keys
        if unknown:
            raise ValueError(
                f"--epoch_param contains invalid key(s) {sorted(unknown)} for "
                f"epoch_method='{epoch_method}'. Valid keys: {sorted(valid_keys)}"
            )

    pipeline_fn, pipeline_key, resolved_pipeline_path = load_pipeline(
        build_params.pipeline,
        build_params.pipeline_params,
    )

    raw_path = dataset_config['raw_path']
    domain_tag = dataset_config.get('domain_tag', dataset_name)
    kernel_spec = dataset_config.get('kernel_spec', None)
    cache_root = os.path.abspath(cache_dir)
    locator_cache_dir = os.path.join(cache_dir, dataset_name)
    locator_info = ensure_locator(
        dataset_name=dataset_name,
        raw_path=raw_path,
        domain_tag=domain_tag,
        kernel_spec=kernel_spec,
        num_workers=build_params.num_workers,
        locator_cache_dir=locator_cache_dir,
        exp_id=build_params.exp_id,
        run_id=build_params.run_id,
    )
    task_locator_info = _apply_task_recording_filter(
        source_locator_info=locator_info,
        task_config=task_config,
        cache_root=cache_root,
        dataset_name=dataset_name,
        task_name=task_name,
    )

    declared_common_channels = (
        task_config.get('common_eeg_channels')
        or dataset_config.get('common_eeg_channels')
    )
    common_channel_info = resolve_common_eeg_channels(
        task_locator_info['path'],
        declared_channels=declared_common_channels,
    )
    channel_layout_token = f"ceegmax_{common_channel_info['signature']}"
    recording_filter_token = _task_recording_filter_token(task_config)
    if recording_filter_token:
        channel_layout_token = f"{channel_layout_token}_rfilter_{recording_filter_token}"
    config_key = compute_config_key(
        epoch_method,
        task_name,
        resolved_epoch_params,
        pipeline_key,
        channel_layout_token=channel_layout_token,
    )

    hdf5_info = _build_hdf5_cache_info(
        cache_root=cache_root,
        dataset_name=dataset_name,
        task_name=task_name,
        raw_path=raw_path,
        domain_tag=domain_tag,
        kernel_spec=kernel_spec,
        config_key=config_key,
        epoch_method=epoch_method,
        pipeline_path=resolved_pipeline_path,
        pipeline_key=pipeline_key,
        pipeline_params=build_params.pipeline_params,
        epoch_params=resolved_epoch_params,
        parent_locator_cache_id=locator_info['cache_id'],
        channel_policy=common_channel_info['policy'],
        common_eeg_channels=common_channel_info['channels'],
        common_eeg_signature=common_channel_info['signature'],
    )

    task_cache_dir = os.path.dirname(hdf5_info['absolute_path'])
    hdf5_path = hdf5_info['absolute_path']
    lock_path = os.path.join(task_cache_dir, f".{config_key}.lock")
    tmp_hdf5_path = f"{hdf5_path}.tmp"

    os.makedirs(task_cache_dir, exist_ok=True)
    with _exclusive_file_lock(lock_path, 'hdf5'):
        should_count_rebuild = False

        if os.path.exists(hdf5_path) and not build_params.rebuild_hdf5:
            is_valid, reason = _validate_hdf5_cache(hdf5_path)
            if is_valid:
                print(f"  [hdf5] Cache hit: {hdf5_path}")
                if os.path.exists(locator_info['absolute_path']):
                    locator_valid, locator_reason = _validate_locator_cache(locator_info['absolute_path'])
                    if locator_valid:
                        _safe_record_cache_event(
                            row_update=_augment_locator_stats(locator_info, locator_info['absolute_path']),
                            event_type="cache_hit",
                            exp_id=build_params.exp_id,
                            run_id=build_params.run_id,
                        )
                    else:
                        _safe_record_cache_event(
                            row_update=_augment_locator_stats(locator_info, locator_info['absolute_path']),
                            event_type="cache_corrupted",
                            exp_id=build_params.exp_id,
                            run_id=build_params.run_id,
                            error_message=locator_reason,
                        )
                else:
                    _safe_record_cache_event(
                        row_update=_apply_file_state(locator_info, locator_info['absolute_path']),
                        event_type="cache_deleted",
                        exp_id=build_params.exp_id,
                        run_id=build_params.run_id,
                        error_message="locator cache missing during hdf5 cache hit",
                    )

                _safe_record_cache_event(
                    row_update=_augment_hdf5_stats(hdf5_info, hdf5_path),
                    event_type="cache_hit",
                    exp_id=build_params.exp_id,
                    run_id=build_params.run_id,
                )
                cache_usage = _build_cache_usage(
                    cache_root=cache_root,
                    dataset_name=dataset_name,
                    task_name=task_name,
                    epoch_method=epoch_method,
                    pipeline_path=resolved_pipeline_path,
                    pipeline_key=pipeline_key,
                    pipeline_params=build_params.pipeline_params,
                    epoch_params=resolved_epoch_params,
                    locator_info=task_locator_info,
                    hdf5_info=hdf5_info,
                    common_channel_info=common_channel_info,
                )
                _update_usage_report(build_params.usage_report, cache_usage)
                return hdf5_path
            print(f"  [hdf5] Corrupted cache detected ({reason}) — rebuilding: {hdf5_path}")
            _safe_record_cache_event(
                row_update=_augment_hdf5_stats(hdf5_info, hdf5_path),
                event_type="cache_corrupted",
                exp_id=build_params.exp_id,
                run_id=build_params.run_id,
                error_message=reason,
            )
            _remove_if_exists(hdf5_path)
            _remove_if_exists(tmp_hdf5_path)
            _safe_record_cache_event(
                row_update=_apply_file_state(hdf5_info, hdf5_path),
                event_type="cache_deleted",
                exp_id=build_params.exp_id,
                run_id=build_params.run_id,
                error_message="removed corrupted hdf5 cache before rebuild",
            )
            should_count_rebuild = True

        if build_params.rebuild_hdf5 and os.path.exists(hdf5_path):
            print(f"  [hdf5] Forcing rebuild — removing: {hdf5_path}")
            _safe_record_cache_event(
                row_update=_apply_file_state(hdf5_info, hdf5_path),
                event_type="cache_deleted",
                exp_id=build_params.exp_id,
                run_id=build_params.run_id,
                error_message="removed existing hdf5 due to --rebuild_hdf5",
            )
            _remove_if_exists(hdf5_path)
            _remove_if_exists(tmp_hdf5_path)
            should_count_rebuild = True

        print(f"  [hdf5] Cache miss — building HDF5 for '{dataset_name}:{task_name}' ...")
        _safe_record_cache_event(
            row_update=_apply_file_state(hdf5_info, hdf5_path),
            event_type="build_started",
            exp_id=build_params.exp_id,
            run_id=build_params.run_id,
            update_registry=False,
        )

        try:
            # 1. Initialize UnifiedDataset from locator
            from eegunity import UnifiedDataset
            u_ds_kwargs = dict(
                locator_path=task_locator_info['path'],
                num_workers=build_params.num_workers,
                verbose='WARNING',
            )
            if kernel_spec is not None:
                u_ds_kwargs['kernel_spec'] = kernel_spec
            u_ds = UnifiedDataset(**u_ds_kwargs)

            # Only write epochs for events declared in event_dict (minimises HDF5 size).
            include_events = list(task_config['event_dict'].keys()) if task_config.get('event_dict') else None

            # 2. Call the appropriate epoch method
            print(f"  [hdf5] Running {epoch_method} (config_key={config_key}) ...")

            if epoch_method == 'epoch_by_event_hdf5':
                tmin = resolved_epoch_params['tmin']
                tmax = resolved_epoch_params['tmax']
                epoch_params = {
                    'tmin': tmin,
                    'tmax': tmax,
                    'baseline': None if tmin >= 0 else (None, 0),
                }
                u_ds.eeg_batch.epoch_by_event_hdf5(
                    output_path=task_cache_dir,
                    file_name_prefix=config_key,
                    miss_bad_data=True,
                    include_events=include_events,
                    resample_params={},
                    epoch_params=epoch_params,
                    pipeline=pipeline_fn,
                    common_eeg_channels=common_channel_info['channels'],
                )

            elif epoch_method == 'epoch_by_segmentation_hdf5':
                segment_params = {
                    'segment_length': resolved_epoch_params['segment_length'],
                    'overlap': resolved_epoch_params['overlap'],
                }
                u_ds.eeg_batch.epoch_by_segmentation_hdf5(
                    output_path=task_cache_dir,
                    file_name_prefix=config_key,
                    miss_bad_data=True,
                    resample_params={},
                    segment_params=segment_params,
                    pipeline=pipeline_fn,
                    common_eeg_channels=common_channel_info['channels'],
                )

            elif epoch_method == 'epoch_by_long_event_hdf5':
                tmin = resolved_epoch_params['tmin']
                tmax = resolved_epoch_params['tmax']
                epoch_params = {
                    'tmin': tmin,
                    'tmax': tmax,
                    'baseline': None if tmin >= 0 else (None, 0),
                    'event_repeated': 'drop',
                }
                u_ds.eeg_batch.epoch_by_long_event_hdf5(
                    output_path=task_cache_dir,
                    file_name_prefix=config_key,
                    overlap=resolved_epoch_params['overlap'],
                    miss_bad_data=True,
                    include_events=include_events,
                    resample_params={},
                    epoch_params=epoch_params,
                    pipeline=pipeline_fn,
                    common_eeg_channels=common_channel_info['channels'],
                )

            else:
                raise ValueError(f"Unknown epoch_method: '{epoch_method}'")

            if not os.path.exists(hdf5_path):
                raise RuntimeError(
                    f"HDF5 build completed but expected output not found: {hdf5_path}\n"
                    f"Check that eegunity wrote to: {task_cache_dir}/{config_key}.hdf5"
                )

            _write_hdf5_channel_metadata(hdf5_path, common_channel_info)
            print(f"  [hdf5] Built successfully: {hdf5_path}")
            _safe_record_cache_event(
                row_update=_augment_hdf5_stats(hdf5_info, hdf5_path),
                event_type="build_succeeded",
                exp_id=build_params.exp_id,
                run_id=build_params.run_id,
                is_rebuild=should_count_rebuild,
            )
            cache_usage = _build_cache_usage(
                cache_root=cache_root,
                dataset_name=dataset_name,
                task_name=task_name,
                epoch_method=epoch_method,
                pipeline_path=resolved_pipeline_path,
                pipeline_key=pipeline_key,
                pipeline_params=build_params.pipeline_params,
                epoch_params=resolved_epoch_params,
                locator_info=task_locator_info,
                hdf5_info=hdf5_info,
                common_channel_info=common_channel_info,
            )
            _update_usage_report(build_params.usage_report, cache_usage)
            return hdf5_path
        except Exception as exc:
            _safe_record_cache_event(
                row_update=_apply_file_state(hdf5_info, hdf5_path),
                event_type="build_failed",
                exp_id=build_params.exp_id,
                run_id=build_params.run_id,
                error_message=str(exc),
            )
            raise
