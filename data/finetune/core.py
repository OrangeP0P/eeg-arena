import os
import sys
import json
import h5py
import numpy as np
import torch
from torch.utils.data import TensorDataset, Subset

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
if MODELS_DIR not in sys.path:
    sys.path.insert(0, MODELS_DIR)

from .dataset import EpochDataset, EpochDatasetWrapper
from .sampling import sample_index_list_by_ratio, validate_data_ratio
from .channel_sparsity import resolve_channel_sparsity_variant
from eegunity.modules.parser.eeg_parser import channel_name_parser
from datasets import get_dataset_config, get_task_config, list_all

# Import channel systems from unified configuration
from data.channels import get_channel_system


_DYNAMIC_CHANNEL_SYSTEM = "dynamic"


def _norm_ch_prefix(name: str) -> str:
    """Normalize the type prefix of a channel name to lowercase.

    Converts e.g. ``'EEG:Fp1'`` → ``'eeg:Fp1'`` and ``'EOG:ch01'`` →
    ``'eog:ch01'`` while leaving the electrode-name portion unchanged.
    This enables case-insensitive prefix matching between eegunity ≤0.5.4
    (uppercase ``'EEG:'``) and ≥0.7.4 (lowercase ``'eeg:'``).
    """
    if ':' in name:
        prefix, rest = name.split(':', 1)
        return prefix.lower() + ':' + rest
    return name


def _canonical_channel_token(name: str) -> str:
    """Case-insensitive token used when matching formatted channel names."""
    normalized = _norm_ch_prefix(str(name)).replace(" ", "")
    if ':' in normalized:
        prefix, rest = normalized.split(':', 1)
        return f"{prefix}:{rest.upper()}"
    return normalized.upper()


def _bipolar_endpoint_tokens(name: str):
    normalized = _norm_ch_prefix(str(name)).replace(" ", "")
    if ':' in normalized:
        prefix, rest = normalized.split(':', 1)
    else:
        prefix, rest = "eeg", normalized

    if '-' not in rest:
        return None

    left, right = rest.split('-', 1)
    if not left or not right:
        return None
    return f"{prefix}:{left.upper()}", f"{prefix}:{right.upper()}"


def _format_channel_preview(channels, limit=8):
    channels = list(channels or [])
    if not channels:
        return "[]"
    preview = ", ".join(str(channel) for channel in channels[:limit])
    if len(channels) > limit:
        preview += f", ... (+{len(channels) - limit} more)"
    return f"[{preview}]"


def _resolve_valid_channels_from_user_channels(user_channels, *, selection_label, model_channel_space=None):
    user_channels = list(user_channels)
    if not user_channels:
        raise ValueError(f"Channel selection '{selection_label}' resolved to an empty user channel list.")

    if model_channel_space is None:
        return list(user_channels)

    model_channels = list(model_channel_space)
    user_channel_tokens = {_canonical_channel_token(channel) for channel in user_channels}
    valid_channels = []
    for channel in model_channels:
        channel_token = _canonical_channel_token(channel)
        if channel_token in user_channel_tokens:
            valid_channels.append(channel)
            continue

        # Some pretrained vocabularies, including FEMBA's TUH TCP montage,
        # contain bipolar names. Keep those when both endpoint electrodes are
        # allowed by the selected channel space, so already-bipolar datasets
        # can be passed through instead of being dropped during channel mapping.
        endpoint_tokens = _bipolar_endpoint_tokens(channel)
        if endpoint_tokens is not None and all(token in user_channel_tokens for token in endpoint_tokens):
            valid_channels.append(channel)

    if not valid_channels:
        raise ValueError(
            "Channel intersection is empty: selected channel space "
            f"'{selection_label}' has no overlap with the model channel space. "
            f"Selected channels ({len(user_channels)}): {_format_channel_preview(user_channels)}. "
            f"Model channels ({len(model_channels)}): {_format_channel_preview(model_channels)}. "
            "Use a compatible channel selection or adjust the model channel-space mapping."
        )

    return valid_channels


def _resolve_valid_channels(channel_system='10-20', model_channel_space=None):
    """
    Resolve the final allowed channels for data loading.

    The user-selected channel system is always respected. When a model declares
    its own pretrained channel space, we further constrain the selection to the
    intersection of the two spaces.
    """
    return _resolve_valid_channels_from_user_channels(
        get_channel_system(channel_system),
        selection_label=f"channel_system:{channel_system}",
        model_channel_space=model_channel_space,
    )


def _json_attr_list(value):
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        return json.loads(value)
    return list(value)


def _load_hdf5_common_eeg_channels(data_path):
    with h5py.File(data_path, 'r') as f:
        common_channels = _json_attr_list(f.attrs.get('common_eeg_channels'))
        if common_channels:
            return list(common_channels)
        channel_names = _json_attr_list(f.attrs.get('ch_names'))
        if channel_names:
            return list(channel_names)
    raise ValueError(
        f"HDF5 file '{data_path}' does not contain 'common_eeg_channels' or 'ch_names' root attrs."
    )


def _load_npy_formatted_channels(data_path):
    channel_list = np.load(f'{data_path}/channel.npy')
    channel_str = ', '.join(channel_list)
    return channel_name_parser(channel_str).split(', ')


def _resolve_user_channel_selection(
    dataset_name,
    task_name,
    data_path,
    *,
    channel_system,
    channel_variant=None,
    channel_count_target=None,
    channel_sparsity_config=None,
):
    if channel_system == 'custom':
        if not channel_variant:
            raise ValueError("channel_system='custom' requires a non-empty channel_variant.")

        if os.path.isfile(data_path):
            available_channels = _load_hdf5_common_eeg_channels(data_path)
        else:
            available_channels = _load_npy_formatted_channels(data_path)

        resolved = resolve_channel_sparsity_variant(
            dataset_name,
            task_name,
            channel_variant,
            available_channels,
            config_path=channel_sparsity_config,
        )
        resolved_target = resolved["target_count"]
        if channel_count_target is not None and int(channel_count_target) != resolved_target:
            raise ValueError(
                f"channel_variant '{channel_variant}' for '{dataset_name}:{task_name}' resolves to "
                f"{resolved_target} channels, but channel_count_target={channel_count_target} was provided."
            )
        return {
            "selection_label": f"channel_sparsity:{resolved['name']}",
            "user_channels": list(resolved["selected_channels"]),
            "channel_count_target": resolved_target,
            "channel_variant": resolved["name"],
            "channel_sparsity_config": resolved.get("config_path"),
        }

    if channel_variant is not None:
        raise ValueError(
            f"channel_variant='{channel_variant}' is only valid when channel_system='custom'."
        )
    if channel_count_target is not None:
        raise ValueError(
            f"channel_count_target={channel_count_target} is only valid when channel_system='custom'."
        )
    return {
        "selection_label": f"channel_system:{channel_system}",
        "user_channels": list(get_channel_system(channel_system)),
        "channel_count_target": None,
        "channel_variant": None,
        "channel_sparsity_config": None,
    }


def _normalize_selected_channel_name(name: str) -> str:
    """Clean display channel names without discarding bipolar derivations."""
    return str(name).strip().split(' ')[-1]


def _resolve_dataset_data_path(dataset_name, user_data_path, ds_config):
    """
    Resolve dataset path with priority:
    1) user-provided path (file or folder)
    2) registry-configured absolute path
    """
    candidates = []
    user_data_path = (user_data_path or "").strip()

    def _looks_like_placeholder(path_str):
        return "set/to/your/datasets" in path_str.replace("\\", "/")

    # User provides a direct HDF5 file
    if user_data_path and not _looks_like_placeholder(user_data_path):
        if os.path.isfile(user_data_path):
            return user_data_path
        candidates.append(user_data_path)

    # User provides a dataset root directory
    if user_data_path and os.path.isdir(user_data_path):
        # 1) If registry has a data_file name, prefer <user_dir>/<data_file>
        if ds_config is not None:
            data_file = ds_config.get("data_file", "")
            if data_file:
                joined = os.path.join(user_data_path, data_file)
                if os.path.exists(joined):
                    return joined
                candidates.append(joined)

        # 2) Support legacy NPY directory format
        npy_required = [
            "train_data.npy", "train_label.npy",
            "val_data.npy", "val_label.npy",
            "test_data.npy", "test_label.npy", "channel.npy",
        ]
        if all(os.path.exists(os.path.join(user_data_path, f)) for f in npy_required):
            return user_data_path

    # Fallback to registry absolute path
    if ds_config is not None:
        folder = ds_config.get("folder", "")
        data_file = ds_config.get("data_file", "")
        if folder and data_file:
            joined = os.path.join(folder, data_file)
            if os.path.exists(joined):
                return joined
            candidates.append(joined)
        elif folder and os.path.exists(folder):
            return folder

    joined_candidates = "\n  - ".join(candidates) if candidates else "(no candidate path)"
    raise FileNotFoundError(
        f"Cannot resolve data path for dataset '{dataset_name}'. Tried:\n  - {joined_candidates}"
    )


def resolve_dynamic_channel_system(
    dataset_name: str,
    task_name: str,
    requested_channel_system: str,
    *,
    ds_config=None,
    task_config=None,
) -> str:
    """Resolve the concrete channel system used by finetune experiments."""
    if requested_channel_system != _DYNAMIC_CHANNEL_SYSTEM:
        return requested_channel_system

    task_config = task_config if task_config is not None else get_task_config(dataset_name, task_name)
    ds_config = ds_config if ds_config is not None else get_dataset_config(dataset_name)

    resolved = None
    if task_config is not None:
        resolved = task_config.get('dynamic_channel_system')
    if resolved is None and ds_config is not None:
        resolved = ds_config.get('dynamic_channel_system')

    if not resolved:
        raise ValueError(
            f"Dataset task '{dataset_name}:{task_name}' does not declare a "
            "'dynamic_channel_system' in the dataset registry, so "
            "--channel_system dynamic cannot be resolved."
        )
    return resolved


def load_channel_positions(ch_names):
    """Load authoritative 3D channel positions for the selected electrodes.

    Standard electrode names are resolved from MNE's ``standard_1005``
    montage.  The historical ``combined_montage.json`` remains a fallback for
    project-specific names, while bipolar channels are placed at the midpoint
    of their two standard endpoints.

    Args:
        ch_names: List of channel names (e.g., ['EEG:Fp1', 'EEG:Fp2', ...] or ['Fp1', 'Fp2', ...])

    Returns:
        dict: Dictionary mapping channel name to 3D position [x, y, z]
    """
    # Remove type prefix (e.g. 'EEG:' or 'eeg:') if present to match montage format
    clean_ch_names = [ch.split(':', 1)[1] if ':' in ch else ch for ch in ch_names]

    # Do not use the legacy JSON for standard names: older generated versions
    # can contain coordinates shifted relative to their keys.  MNE is already
    # a required EEG-Arena dependency and is the source used by dataset kernels.
    standard_positions = {}
    try:
        import mne

        montage = mne.channels.make_standard_montage("standard_1005")
        standard_positions = {
            name.upper(): [float(value) for value in position]
            for name, position in montage.get_positions()["ch_pos"].items()
        }
    except Exception as exc:
        print(f"Warning: Could not load MNE standard_1005 montage: {exc}")

    montage_path = os.path.join(PROJECT_ROOT, "combined_montage.json")
    montage_data = {}
    if os.path.exists(montage_path):
        with open(montage_path, 'r') as f:
            montage_data = json.load(f)

    fallback_positions = {
        str(name).upper(): [float(value) for value in position]
        for name, position in montage_data.items()
    }

    def _standard_position(name):
        token = str(name).replace(" ", "").strip().upper()
        direct = standard_positions.get(token)
        if direct is not None:
            return direct
        if "-" in token:
            left, right = token.split("-", 1)
            left_pos = standard_positions.get(left)
            right_pos = standard_positions.get(right)
            if left_pos is not None and right_pos is not None:
                return [
                    (left_value + right_value) / 2.0
                    for left_value, right_value in zip(left_pos, right_pos)
                ]
        return None

    ch_positions = {}
    for i, ch in enumerate(ch_names):
        clean_ch = clean_ch_names[i]
        position = _standard_position(clean_ch)
        if position is not None:
            ch_positions[ch] = position
        elif clean_ch.upper() in fallback_positions:
            ch_positions[ch] = fallback_positions[clean_ch.upper()]
        else:
            print(f"Warning: Channel {clean_ch} not found in montages, setting to [0, 0, 0]")
            ch_positions[ch] = [0.0, 0.0, 0.0]
    return ch_positions

def _build_channel_mapping(dataset_name, data_path, event_dict, *, valid_channels, selection_label):
    """Build channel mapping for specified channel system

    Args:
        dataset_name: Name of the dataset
        data_path: Path to the data
        event_dict: Event dictionary
        valid_channels: Final channel list after applying the user space and
            optional model-space intersection.
        selection_label: Human-readable label used in error messages.
    """
    # Build case-insensitive lookup: canonical token → canonical display name.
    norm_to_canonical = {_canonical_channel_token(ch): ch for ch in valid_channels}

    with h5py.File(data_path, 'r') as f:
        version = f.attrs.get('version', None)
        if version != '2.0':
            raise ValueError(
                f"HDF5 file '{data_path}' is v1 format (version={version!r}), "
                "which is no longer supported. Delete the cached file and re-run "
                "with --rebuild_hdf5 to generate a v2 file."
            )
        channel_list = json.loads(f.attrs['ch_names'])

    channel_str = ', '.join(channel_list)
    format_channel = channel_name_parser(channel_str)
    format_channel_list = format_channel.split(', ')

    # Build normalized mapping: formatted token → raw HDF5 name.
    norm_channel_mapping = {
        _canonical_channel_token(fmt): raw
        for fmt, raw in zip(format_channel_list, channel_list)
    }

    modified_channel_list = []     # canonical names (from data/channels.py)
    original_corresponding_list = []  # raw HDF5 names (used to index into HDF5)
    for fmt_ch in format_channel_list:
        norm_ch = _canonical_channel_token(fmt_ch)
        if norm_ch in norm_to_canonical:
            modified_channel_list.append(norm_to_canonical[norm_ch])
            original_corresponding_list.append(norm_channel_mapping[norm_ch])

    if not original_corresponding_list:
        raise ValueError(
            "Channel intersection is empty for "
            f"{dataset_name} HDF5 file '{data_path}': resolved selection "
            f"'{selection_label}' produced {len(valid_channels)} candidate channels, "
            "but none matched the HDF5 channel names. "
            f"Candidate channels: {_format_channel_preview(valid_channels)}. "
            f"HDF5 channels ({len(channel_list)}): {_format_channel_preview(channel_list)}. "
            "Check the dataset dynamic_channel_system/common_eeg_channels settings, "
            "or rebuild the HDF5 cache if it is stale."
        )

    updated_channel_list = list(modified_channel_list)

    return original_corresponding_list, updated_channel_list


def _sorted_unique_subjects_from_values(subject_values):
    unique_subjects = np.unique(subject_values)
    unique_subjects.sort()
    return [str(subject) for subject in unique_subjects.tolist()]


def _get_sorted_unique_subjects(dataset):
    return _sorted_unique_subjects_from_values(dataset.get_subjects())


def _subject_split_metadata(*, train_subjects=None, val_subjects=None, test_subjects=None):
    def _normalize(subjects):
        if subjects is None:
            return None
        return [str(subject) for subject in subjects]

    return {
        "train_subjects": _normalize(train_subjects),
        "val_subjects": _normalize(val_subjects),
        "test_subjects": _normalize(test_subjects),
    }


def _normalize_preset_split_name(value):
    text = str(value).strip().lower()
    aliases = {
        "train": "train",
        "training": "train",
        "val": "val",
        "valid": "val",
        "validation": "val",
        "dev": "val",
        "test": "test",
        "testing": "test",
    }
    return aliases.get(text)


def _sample_split_indices(split_indices, *, data_ratio, sampling_seed, context_parts):
    sampled_indices = {}
    sample_counts = {}
    for split_name, index_list in split_indices.items():
        split_sampled_indices, split_counts = sample_index_list_by_ratio(
            index_list,
            data_ratio,
            seed=sampling_seed,
            split_name=split_name,
            context_parts=context_parts,
        )
        sampled_indices[split_name] = split_sampled_indices
        sample_counts[f"{split_name}_original"] = split_counts["original"]
        sample_counts[split_name] = split_counts["sampled"]
    return sampled_indices, sample_counts


def resolve_loso_target(subject_ids, fold=1, held_out_subject=None):
    if not subject_ids:
        raise ValueError("LOSO split requires at least one subject.")

    normalized_subjects = [str(subject) for subject in subject_ids]
    if held_out_subject is not None:
        resolved_subject = str(held_out_subject).strip()
        if not resolved_subject:
            raise ValueError("held_out_subject must be a non-empty string when provided.")
        if resolved_subject not in normalized_subjects:
            raise ValueError(
                f"held_out_subject '{resolved_subject}' not found. "
                f"Available subjects: {normalized_subjects}"
            )
        return normalized_subjects.index(resolved_subject) + 1, resolved_subject

    if fold is None:
        raise ValueError("LOSO split requires either fold or held_out_subject.")
    resolved_subject = normalized_subjects[(int(fold) - 1) % len(normalized_subjects)]
    return normalized_subjects.index(resolved_subject) + 1, resolved_subject


def load_data(
    dataset_name,
    task_name,
    data_path,
    event_dict,
    fold,
    split_mode="cs",
    channel_system='10-20',
    channel_variant=None,
    channel_count_target=None,
    channel_sparsity_config=None,
    patch_size=200,
    input_format='patched',
    model_channel_space=None,
    held_out_subject=None,
    data_ratio=1.0,
    sampling_seed=0,
):
    """Load dataset with specified channel system

    Args:
        dataset_name: Name of the dataset
        task_name: Task name within the dataset
        data_path: Path to the data
        event_dict: Event dictionary
        fold: Fold index
        split_mode: 'cs' (cross-subject), 'ws' (within-subject), 'loso',
            or 'cross_trial'
        channel_system: Concrete channel system name already resolved for
            finetune loading.
        channel_variant: Optional channel sparsity subset identifier used only
            when channel_system == 'custom'.
        channel_count_target: Optional expected user-space channel count for
            custom channel variants.
        channel_sparsity_config: Optional JSON file that defines custom channel
            sparsity variants.
        patch_size: Patch size for padding (default 200)
        input_format: 'patched' or 'raw' - only apply padding for 'patched'
        model_channel_space: Model-specific channel list. When provided, it is
            intersected with the user-selected channel_system.
        held_out_subject: Optional subject id used only for LOSO. When set, it
            takes precedence over the fold-to-subject mapping.
        data_ratio: Fraction of each split to keep, sampled deterministically.
        sampling_seed: User seed combined with dataset/split context.

    Returns:
        train_dataset, val_dataset, test_dataset, ch_names, ch_positions,
        num_classes, num_channels, model_channel_indices, split_subjects,
        sample_counts, channel_count_target
    """
    if split_mode not in {"cs", "ws", "loso", "cross_trial"}:
        raise ValueError("split_mode must be 'cs', 'ws', 'loso', or 'cross_trial'")

    data_ratio = validate_data_ratio(data_ratio)
    selection = _resolve_user_channel_selection(
        dataset_name,
        task_name,
        data_path,
        channel_system=channel_system,
        channel_variant=channel_variant,
        channel_count_target=channel_count_target,
        channel_sparsity_config=channel_sparsity_config,
    )
    valid_channels = _resolve_valid_channels_from_user_channels(
        selection["user_channels"],
        selection_label=selection["selection_label"],
        model_channel_space=model_channel_space,
    )

    fold_idx = fold - 1 if split_mode in {"cs", "ws"} else None
    num_classes = None
    split_subjects = _subject_split_metadata()
    sample_counts = {}
    sampling_context_parts = (
        dataset_name,
        task_name,
        split_mode,
        fold,
        held_out_subject,
    )

    def _pad_to_multiple(x, patch_size):
        """Pad data to make time length divisible by patch_size"""
        time_len = x.shape[-1]
        pad_len = (patch_size - time_len % patch_size) % patch_size
        if pad_len > 0:
            # Pad with zeros on the last dimension (time)
            padding = [(0, 0)] * (x.ndim - 1) + [(0, pad_len)]
            x = np.pad(x, padding, mode='constant', constant_values=0)
        return x

    if event_dict is not None:
        original_corresponding_list, updated_channel_list = _build_channel_mapping(
            dataset_name,
            data_path,
            event_dict,
            valid_channels=valid_channels,
            selection_label=selection["selection_label"],
        )

        # Create dataset - HDF5 file stays open for efficient access
        # Padding is done inside Dataset when input_format == 'patched'
        pad_to_multiple = patch_size if input_format == 'patched' else None
        dataset = EpochDataset(
            data_path,
            event_dict,
            channels_select=original_corresponding_list,
            extra_info=['chOrder'],
            dataset_name=dataset_name,
            pad_to_multiple=pad_to_multiple,
        )

        num_classes = len(event_dict)
        num_samples = len(dataset)
        split_indices = None

        if split_mode == "cs":
            # Get subjects efficiently without loading data
            all_subjects = dataset.get_subjects()

            unique_subjects = _sorted_unique_subjects_from_values(all_subjects)
            num_subjects = len(unique_subjects)

            # For small subject counts, adjust folds to ensure valid splits
            if num_subjects < 3:
                # Very few subjects: use LOSO with val taken from remaining
                if num_subjects == 2:
                    # 2 subjects: fold 1 uses A as test, B as val; fold 2 uses B as test, A as val
                    fold_idx_adjusted = fold_idx % 2
                    test_subjects = [unique_subjects[fold_idx_adjusted]]
                    val_subjects = [unique_subjects[(fold_idx_adjusted + 1) % 2]]
                    train_subjects = []  # Will be handled below
                else:
                    # 1 subject: cannot do cs split meaningfully
                    raise ValueError(
                        f"Cross-subject split requires at least 2 subjects, but only {num_subjects} found. "
                        f"Consider using 'loso' split_mode instead."
                    )
            elif num_subjects < 5:
                # 3-4 subjects: dynamic folds
                num_folds = num_subjects
                fold_sizes = np.full(num_folds, num_subjects // num_folds, dtype=int)
                fold_sizes[:num_subjects % num_folds] += 1
                current = 0
                subject_folds = []
                for size in fold_sizes:
                    subject_folds.append(unique_subjects[current: current + size])
                    current += size

                test_subjects = subject_folds[fold_idx % num_folds]
                val_subjects = subject_folds[(fold_idx + 1) % num_folds]
                train_subjects = [s for s in unique_subjects if (s not in test_subjects and s not in val_subjects)]
            else:
                # 5+ subjects: standard 5-fold cs split
                num_folds = 5
                fold_sizes = np.full(5, num_subjects // 5, dtype=int)
                fold_sizes[:num_subjects % 5] += 1
                current = 0
                subject_folds = []
                for size in fold_sizes:
                    subject_folds.append(unique_subjects[current: current + size])
                    current += size

                test_subjects = subject_folds[fold_idx % 5]
                val_subjects = subject_folds[(fold_idx + 1) % 5]
                train_subjects = [s for s in unique_subjects if (s not in test_subjects and s not in val_subjects)]

            # For 2-subject case, train comes from remaining subjects after test+val
            if not train_subjects and num_subjects == 2:
                # When only 2 subjects, one is test, one is val, train is empty
                # This is a known limitation of cs split with very few subjects
                pass

            # Build indices for each split
            train_indices = [i for i, s in enumerate(all_subjects) if s in train_subjects]
            val_indices = [i for i, s in enumerate(all_subjects) if s in val_subjects]
            test_indices = [i for i, s in enumerate(all_subjects) if s in test_subjects]

            split_indices = {
                "train": train_indices,
                "val": val_indices,
                "test": test_indices,
            }
            split_subjects = _subject_split_metadata(
                train_subjects=train_subjects,
                val_subjects=val_subjects,
                test_subjects=test_subjects,
            )
        elif split_mode == "ws":
            # Within-subject: random fold split
            # Shuffle indices before splitting
            unique_subjects = _sorted_unique_subjects_from_values(dataset.get_subjects())
            indices = np.arange(num_samples)
            np.random.shuffle(indices)

            fold_sizes = np.full(5, num_samples // 5, dtype=int)
            fold_sizes[:num_samples % 5] += 1
            current = 0
            folds = []
            for fold_size in fold_sizes:
                folds.append(indices[current:current + fold_size])
                current += fold_size

            test_fold = fold_idx
            val_fold = (fold_idx + 1) % 5
            train_folds = [i for i in range(5) if i != test_fold and i != val_fold]

            train_indices = np.concatenate([folds[i] for i in train_folds]).tolist()
            val_indices = folds[val_fold].tolist()
            test_indices = folds[test_fold].tolist()

            split_indices = {
                "train": train_indices,
                "val": val_indices,
                "test": test_indices,
            }
            split_subjects = _subject_split_metadata(
                train_subjects=unique_subjects,
                val_subjects=unique_subjects,
                test_subjects=unique_subjects,
            )
        elif split_mode == "loso":
            # Leave-One-Subject-Out: 每次留一个被试作为测试集
            all_subjects = dataset.get_subjects()

            unique_subjects = _sorted_unique_subjects_from_values(all_subjects)
            _resolved_fold, test_subject = resolve_loso_target(
                unique_subjects,
                fold=fold,
                held_out_subject=held_out_subject,
            )

            # 其余被试用于训练，取一个作为验证集
            remaining_subjects = [s for s in unique_subjects if s != test_subject]
            test_subjects = [test_subject]
            val_subjects = remaining_subjects[:1]
            train_subjects = remaining_subjects[1:]

            # Build indices for each split
            train_indices = [i for i, s in enumerate(all_subjects) if s in train_subjects]
            val_indices = [i for i, s in enumerate(all_subjects) if s in val_subjects]
            test_indices = [i for i, s in enumerate(all_subjects) if s in test_subjects]

            split_indices = {
                "train": train_indices,
                "val": val_indices,
                "test": test_indices,
            }
            split_subjects = _subject_split_metadata(
                train_subjects=train_subjects,
                val_subjects=val_subjects,
                test_subjects=test_subjects,
            )
        elif split_mode == "cross_trial":
            if fold is not None and int(fold) != 1:
                raise ValueError("split_mode='cross_trial' currently supports only fold=1.")
            preset_splits = dataset.get_epoch_meta('preset_split')
            if not dataset.has_epoch_meta('preset_split'):
                raise ValueError(
                    f"Dataset '{dataset_name}' does not provide epoch_meta/preset_split, "
                    "so split_mode='cross_trial' is unavailable."
                )

            normalized_splits = [_normalize_preset_split_name(value) for value in preset_splits]
            invalid_values = sorted({
                str(raw_value)
                for raw_value, normalized in zip(preset_splits, normalized_splits)
                if normalized is None
            })
            if invalid_values:
                raise ValueError(
                    "epoch_meta/preset_split contains unsupported values: "
                    + ", ".join(invalid_values)
                )

            train_indices = [idx for idx, split_name in enumerate(normalized_splits) if split_name == "train"]
            val_indices = [idx for idx, split_name in enumerate(normalized_splits) if split_name == "val"]
            test_indices = [idx for idx, split_name in enumerate(normalized_splits) if split_name == "test"]

            if not train_indices or not val_indices or not test_indices:
                raise ValueError(
                    "split_mode='cross_trial' requires non-empty train/val/test partitions in epoch_meta/preset_split."
                )

            unique_subjects = _sorted_unique_subjects_from_values(dataset.get_subjects())
            split_indices = {
                "train": train_indices,
                "val": val_indices,
                "test": test_indices,
            }
            split_subjects = _subject_split_metadata(
                train_subjects=unique_subjects,
                val_subjects=unique_subjects,
                test_subjects=unique_subjects,
            )

        sampled_split_indices, sample_counts = _sample_split_indices(
            split_indices,
            data_ratio=data_ratio,
            sampling_seed=sampling_seed,
            context_parts=sampling_context_parts,
        )
        train_dataset = EpochDatasetWrapper(Subset(dataset, sampled_split_indices["train"]))
        val_dataset = EpochDatasetWrapper(Subset(dataset, sampled_split_indices["val"]))
        test_dataset = EpochDatasetWrapper(Subset(dataset, sampled_split_indices["test"]))

    else:
        train_data = np.load(f'{data_path}/train_data.npy')
        train_label = np.load(f'{data_path}/train_label.npy')
        validation_data = np.load(f'{data_path}/val_data.npy')
        validation_label = np.load(f'{data_path}/val_label.npy')
        test_data = np.load(f'{data_path}/test_data.npy')
        test_label = np.load(f'{data_path}/test_label.npy')
        num_classes = int(
            len(
                np.unique(
                    np.concatenate([train_label, validation_label, test_label], axis=0)
                )
            )
        )

        channel_list = np.load(f'{data_path}/channel.npy')
        channel_str = ', '.join(channel_list)
        format_channel = channel_name_parser(channel_str)
        format_channel_list = format_channel.split(', ')

        norm_to_canonical_npy = {_canonical_channel_token(ch): ch for ch in valid_channels}
        keep_indices = []
        modified_channel_list = []

        for idx, channel in enumerate(format_channel_list):
            norm_ch = _canonical_channel_token(channel)
            if norm_ch in norm_to_canonical_npy:
                keep_indices.append(idx)
                modified_channel_list.append(norm_to_canonical_npy[norm_ch])

        if not modified_channel_list:
            raise ValueError(
                "Channel intersection is empty for "
                f"{dataset_name} NPY data path '{data_path}': resolved selection "
                f"'{selection['selection_label']}' produced {len(valid_channels)} candidate channels, "
                "but none matched the NPY channel names. "
                f"Candidate channels: {_format_channel_preview(valid_channels)}. "
                f"NPY channels ({len(channel_list)}): {_format_channel_preview(channel_list)}. "
                "Check the selected channel subset or dataset channel metadata."
            )

        keep_indices = np.array(keep_indices)

        train_data = train_data[:, keep_indices, :]
        validation_data = validation_data[:, keep_indices, :]
        test_data = test_data[:, keep_indices, :]

        # Pad data to be divisible by patch_size (only for patched format)
        if input_format == 'patched':
            train_data = _pad_to_multiple(train_data, patch_size)
            validation_data = _pad_to_multiple(validation_data, patch_size)
            test_data = _pad_to_multiple(test_data, patch_size)

        updated_channel_list = list(modified_channel_list)

        sampled_split_indices, sample_counts = _sample_split_indices(
            {
                "train": range(len(train_data)),
                "val": range(len(validation_data)),
                "test": range(len(test_data)),
            },
            data_ratio=data_ratio,
            sampling_seed=sampling_seed,
            context_parts=sampling_context_parts,
        )
        train_take = np.asarray(sampled_split_indices["train"], dtype=int)
        val_take = np.asarray(sampled_split_indices["val"], dtype=int)
        test_take = np.asarray(sampled_split_indices["test"], dtype=int)
        train_data = train_data[train_take]
        train_label = train_label[train_take]
        validation_data = validation_data[val_take]
        validation_label = validation_label[val_take]
        test_data = test_data[test_take]
        test_label = test_label[test_take]

        train_dataset = TensorDataset(torch.from_numpy(train_data).float(), torch.from_numpy(train_label))
        val_dataset = TensorDataset(torch.from_numpy(validation_data).float(), torch.from_numpy(validation_label))
        test_dataset = TensorDataset(torch.from_numpy(test_data).float(), torch.from_numpy(test_label))

    ch_names = updated_channel_list
    ch_names = [_normalize_selected_channel_name(name) for name in ch_names]
    num_channels = len(ch_names)

    # Load channel positions from montage
    ch_positions = load_channel_positions(ch_names)

    # Compute model channel indices (position of each selected channel within the model's channel space)
    model_channel_indices = None
    if model_channel_space is not None:
        norm_model_space = {}
        for i, ch in enumerate(model_channel_space):
            norm_model_space.setdefault(_canonical_channel_token(ch), i)
            norm_model_space.setdefault(_canonical_channel_token(_normalize_selected_channel_name(ch)), i)
        model_channel_indices = []
        for ch in ch_names:
            idx = norm_model_space.get(_canonical_channel_token(ch))
            if idx is not None:
                model_channel_indices.append(idx)

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        ch_names,
        ch_positions,
        num_classes,
        num_channels,
        model_channel_indices,
        split_subjects,
        sample_counts,
        selection["channel_count_target"],
    )


def get_dataset(
    dataset_name,
    task_name,
    fold,
    split_mode="cs",
    channel_system='10-20',
    channel_variant=None,
    channel_count_target=None,
    channel_sparsity_config=None,
    patch_size=200,
    input_format='patched',
    model_channel_space=None,
    build_params=None,
    cache_dir=None,
    held_out_subject=None,
    data_ratio=1.0,
    sampling_seed=0,
):
    """
    Load dataset using the dynamic registry.

    Args:
        dataset_name: Registered dataset name (e.g. 'bcic_iv_2a')
        task_name: Task name within the dataset (e.g. 'motor_imagery')
        fold: fold index
        split_mode: 'cs' (cross-subject), 'ws' (within-subject), 'loso',
            or 'cross_trial'
        channel_system: Public channel system request
            ('10-20', '10-10', '10-5', 'dynamic', or 'custom'). 'dynamic' is resolved
            only for finetune/experiment paths via the dataset registry.
        channel_variant: Optional channel sparsity subset identifier used only
            when channel_system == 'custom'.
        channel_count_target: Optional expected user-space channel count for
            custom channel variants.
        channel_sparsity_config: Optional JSON file that defines custom channel
            sparsity variants.
        patch_size: Patch size for padding (default 200)
        input_format: 'patched' or 'raw'
        model_channel_space: Model-specific channel list
        build_params: HDF5BuildParams instance controlling preprocessing and cache key.
        cache_dir: Base cache directory for HDF5 files and locator CSVs.
        held_out_subject: Optional LOSO subject id. When set, it is resolved to
            the stable fold index for the current dataset and used as the test
            subject.
        data_ratio: Fraction of each split to keep, sampled deterministically.
        sampling_seed: User seed combined with dataset/split context.

    Returns:
        train_dataset, val_dataset, test_dataset, ch_names, ch_positions,
        num_classes, num_channels, model_channel_indices, split_subjects,
        sample_counts, channel_count_target
    """
    ds_config = get_dataset_config(dataset_name)
    if ds_config is None:
        raise ValueError(
            f"Dataset '{dataset_name}' not found in DATASET_REGISTRY. "
            f"Available datasets: {list_all()}"
        )

    task_config = get_task_config(dataset_name, task_name)
    if task_config is None:
        from datasets import list_tasks
        raise ValueError(
            f"Task '{task_name}' not found for dataset '{dataset_name}'. "
            f"Available tasks: {list_tasks(dataset_name)}"
        )

    if cache_dir is None:
        raise ValueError("cache_dir is required for dynamic datasets. Pass --cache_dir.")

    resolved_channel_system = resolve_dynamic_channel_system(
        dataset_name,
        task_name,
        channel_system,
        ds_config=ds_config,
        task_config=task_config,
    )

    from data.raw_to_hdf5 import ensure_hdf5, HDF5BuildParams
    if build_params is None:
        build_params = HDF5BuildParams()

    resolved_hdf5_path = ensure_hdf5(
        dataset_name=dataset_name,
        task_name=task_name,
        dataset_config=ds_config,
        task_config=task_config,
        build_params=build_params,
        cache_dir=cache_dir,
    )
    event_dict = task_config['event_dict']
    train_dataset, val_dataset, test_dataset, ch_names, ch_positions, num_classes, num_channels, model_channel_indices, split_subjects, sample_counts, resolved_channel_count_target = load_data(
        dataset_name, task_name, resolved_hdf5_path, event_dict, fold,
        split_mode=split_mode, channel_system=resolved_channel_system,
        channel_variant=channel_variant, channel_count_target=channel_count_target,
        channel_sparsity_config=channel_sparsity_config,
        patch_size=patch_size, input_format=input_format,
        model_channel_space=model_channel_space,
        held_out_subject=held_out_subject,
        data_ratio=data_ratio,
        sampling_seed=sampling_seed,
    )
    return (
        train_dataset,
        val_dataset,
        test_dataset,
        ch_names,
        ch_positions,
        num_classes,
        num_channels,
        model_channel_indices,
        split_subjects,
        sample_counts,
        resolved_channel_count_target,
    )


def get_subject_ids(dataset_name, task_name, build_params=None, cache_dir=None):
    """Return the sorted unique subject ids for one dataset task."""
    ds_config = get_dataset_config(dataset_name)
    task_config = get_task_config(dataset_name, task_name)

    if ds_config is None or task_config is None:
        raise ValueError(f"Unknown dataset/task: '{dataset_name}:{task_name}'")

    if cache_dir is None:
        raise ValueError("cache_dir is required. Pass --cache_dir.")

    from data.raw_to_hdf5 import ensure_hdf5, HDF5BuildParams
    if build_params is None:
        build_params = HDF5BuildParams()

    data_path = ensure_hdf5(
        dataset_name=dataset_name,
        task_name=task_name,
        dataset_config=ds_config,
        task_config=task_config,
        build_params=build_params,
        cache_dir=cache_dir,
    )
    event_dict = task_config['event_dict']

    dataset = EpochDataset(
        data_path,
        event_dict,
        channels_select=None,
        extra_info=['chOrder'],
        dataset_name=dataset_name,
    )
    return _get_sorted_unique_subjects(dataset)


def resolve_loso_fold_and_subject(
    dataset_name,
    task_name,
    *,
    fold=1,
    held_out_subject=None,
    build_params=None,
    cache_dir=None,
):
    subject_ids = get_subject_ids(
        dataset_name,
        task_name,
        build_params=build_params,
        cache_dir=cache_dir,
    )
    return resolve_loso_target(
        subject_ids,
        fold=fold,
        held_out_subject=held_out_subject,
    )


def get_num_subjects(dataset_name, task_name, fold=1, split_mode="cs", channel_system='10-20', build_params=None, cache_dir=None):
    """
    Get the number of subjects in the dataset.

    Args:
        dataset_name: Registered dataset name
        task_name: Task name within the dataset
        fold: fold index (unused for subject counting, kept for API consistency)
        split_mode: split mode (unused here)
        channel_system: channel system (unused here)
        build_params: HDF5BuildParams for task datasets (optional)
        cache_dir: Base cache directory for HDF5 files

    Returns:
        int: Number of unique subjects
    """
    return len(
        get_subject_ids(
            dataset_name,
            task_name,
            build_params=build_params,
            cache_dir=cache_dir,
        )
    )
