"""
Dataset Registry - Unified dataset configuration for dynamic EEG datasets.

Registration mode:
  register_dataset_dynamic(name, config)  — builds HDF5 from raw data on demand

Config structure:
  dataset-level: raw_path, domain_tag, kernel_spec, subject_parser,
                 dynamic_channel_system, common_eeg_channels
  task-level (inside config['tasks']): epoch_method, event_dict,
                                        default_epoch_params, num_classes, task_type,
                                        recording_filter, dynamic_channel_system,
                                        common_eeg_channels

CLI usage:
  --dataset <dataset_name> --task <task_name>
"""
import os

# Project root = parent of this file (EEG-Arena/)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_KERNELS_DIR = os.path.join(_PROJECT_ROOT, 'datasets', 'kernels')
_DEFAULT_RAW_DATA_ROOT = os.path.join(_PROJECT_ROOT, 'data', 'raw')
_RAW_DATA_ROOT = os.path.abspath(os.path.expanduser(
    os.environ.get('EEG_ARENA_RAW_DATA_ROOT')
    or os.environ.get('EEG2_RAW_DATA_ROOT')
    or _DEFAULT_RAW_DATA_ROOT
))


def _resolve_raw_path(raw_path):
    """Resolve registry-relative paths under the configurable raw-data root."""
    if not isinstance(raw_path, str):
        return raw_path

    expanded = os.path.expanduser(raw_path)
    if os.path.isabs(expanded):
        return os.path.normpath(expanded)
    return os.path.normpath(os.path.join(_RAW_DATA_ROOT, expanded))

# ============ Registry ============
# dataset_name -> {raw_path, domain_tag, kernel_spec, subject_parser, tasks: {task_name: {...}}}
DATASET_REGISTRY = {}


def register_dataset_dynamic(name, config):
    """
    Register a dynamic dataset (HDF5 built on demand from raw data).

    Args:
        name: dataset name (lowercase, unique)
        config: dict with keys:
            raw_path        (str, required)  — path to the raw EEG dataset directory
            domain_tag      (str, required)  — domain tag passed to UnifiedDataset
            kernel_spec     (str|None)       — path to kernel .py; None = generic BIDS
            subject_parser  (dict, optional) — how to parse subject id from group name:
                {"type": "slice", "start": 0, "end": 3}
                {"type": "split", "sep": "_", "index": 0}
                {"type": "regex", "pattern": "(Subject\\d+)", "group": 1}
            tasks           (dict, required) — task_name -> task_config dict

        Each task_config has keys:
            epoch_method    (str, required)  — one of:
                                               'epoch_by_event_hdf5'
                                               'epoch_by_segmentation_hdf5'
                                               'epoch_by_long_event_hdf5'
            default_epoch_params (dict)      — default tmin/tmax/segment_length/overlap
                                               (overridable via --epoch_param)
            event_dict      (dict, required) — event name -> integer label mapping
            num_classes     (int, optional)  — inferred from event_dict if omitted
            task_type       (str, optional)  — 'multiclass', 'binary', or 'regression'
            recording_filter (dict, optional) — {"basename_regex": "...",
                                                 "required_eeg_channels": ["..."]}
            dynamic_channel_system (str, optional) — registry-only concrete
                                                     system used when the user
                                                     requests --channel_system dynamic

    Example:
        register_dataset_dynamic('bcic_iv_2a', {
            'raw_path':     'bcic_iv_2a/',
            'domain_tag':   'bcic_iv_2a',
            'kernel_spec':  '/path/to/bcic_iv_2a.py',
            'subject_parser': {'type': 'slice', 'start': 0, 'end': 3},
            'tasks': {
                'motor_imagery': {
                    'epoch_method': 'epoch_by_event_hdf5',
                    'default_epoch_params': {'tmin': -0.5, 'tmax': 4.0},
                    'event_dict': {'left': 0, 'right': 1},
                    'num_classes': 2,
                    'task_type': 'binary',
                },
            },
        })
    """
    normalized = dict(config)
    raw_path = normalized.get('raw_path')
    normalized['raw_path'] = _resolve_raw_path(raw_path)
    DATASET_REGISTRY[name.lower()] = normalized


def get_dataset_config(name):
    """Get dataset-level config (without tasks dict) by dataset name."""
    cfg = DATASET_REGISTRY.get(name.lower())
    if cfg is None:
        return None
    return {k: v for k, v in cfg.items() if k != 'tasks'}


def get_task_config(dataset_name, task_name):
    """Get task-level config by dataset and task name.

    Returns None if dataset or task is not found.
    """
    cfg = DATASET_REGISTRY.get(dataset_name.lower())
    if cfg is None:
        return None
    return cfg.get('tasks', {}).get(task_name.lower())


def list_datasets():
    """List all registered dataset names."""
    return list(DATASET_REGISTRY.keys())


def list_tasks(dataset_name):
    """List all task names for a given dataset."""
    cfg = DATASET_REGISTRY.get(dataset_name.lower())
    if cfg is None:
        return []
    return list(cfg.get('tasks', {}).keys())


def list_all():
    """List all (dataset, task) pairs as 'dataset:task' strings."""
    result = []
    for ds_name, cfg in DATASET_REGISTRY.items():
        for task_name in cfg.get('tasks', {}).keys():
            result.append(f"{ds_name}:{task_name}")
    return result


_EVENT_HDF5 = 'epoch_by_event_hdf5'
_LONG_EVENT_HDF5 = 'epoch_by_long_event_hdf5'


def _task(epoch_method, default_epoch_params, event_dict, num_classes, task_type,
          recording_filter=None, dynamic_channel_system=None, common_eeg_channels=None):
    config = {
        'epoch_method': epoch_method,
        'default_epoch_params': default_epoch_params,
        'event_dict': event_dict,
        'num_classes': num_classes,
        'task_type': task_type,
    }
    if recording_filter is not None:
        config['recording_filter'] = recording_filter
    if dynamic_channel_system is not None:
        config['dynamic_channel_system'] = dynamic_channel_system
    if common_eeg_channels is not None:
        config['common_eeg_channels'] = list(common_eeg_channels)
    return config


_CAPSLPDB_DYNAMIC_CHANNEL_SYSTEM = 'capslpdb-5ch'
_CAPSLPDB_REQUIRED_EEG_CHANNELS = {
    "required_eeg_channels": ["Fp2-F4", "F4-C4", "C4-P4", "P4-O2", "C4-A1"],
}

_CHBMIT_DYNAMIC_CHANNEL_SYSTEM = 'chbmit-bipolar-17ch'
_CHBMIT_REQUIRED_EEG_CHANNELS = {
    "required_eeg_channels": [
        "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
        "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
        "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
        "FP2-F8", "F8-T8", "FZ-CZ", "CZ-PZ", "P8-O2",
    ],
}

_HMCSLEEP_DYNAMIC_CHANNEL_SYSTEM = 'hmcsleep-4ch'
_SLEEPEDFX_DYNAMIC_CHANNEL_SYSTEM = 'sleepedfx-2ch'
_UCDDB_DYNAMIC_CHANNEL_SYSTEM = 'ucddb-2ch'


# ============ Register built-in dynamic datasets ============

# BCIC IV 1 - subject-specific 2-class motor imagery tasks.
register_dataset_dynamic('bcic_iv_1', {
    'raw_path':    'bcic_iv_1/',
    'domain_tag':  'bcic_iv_1',
    'kernel_spec': os.path.join(_KERNELS_DIR, 'bcic_iv_1.py'),
    'subject_parser': {"type": "regex", "pattern": "(ds1[a-z])", "group": 1},
    'tasks': {
        'motor_imagery_left_right_2class': _task(
            _EVENT_HDF5, {'tmin': 0.5, 'tmax': 2.5},
            {"left": 0, "right": 1},
            2, 'binary',
            recording_filter={
                "basename_regex": r"^BCICIV_calib_ds1[bcdeg]_1000Hz\.mat$",
            },
        ),
        'motor_imagery_left_foot_2class': _task(
            _EVENT_HDF5, {'tmin': 0.5, 'tmax': 2.5},
            {"left": 0, "foot": 1},
            2, 'binary',
            recording_filter={
                "basename_regex": r"^BCICIV_calib_ds1[af]_1000Hz\.mat$",
            },
        ),
    },
})

# BCIC IV 2a - 4-class and standard binary motor imagery variants.
register_dataset_dynamic('bcic_iv_2a', {
    'raw_path':    'bcic_iv_2a/',
    'domain_tag':  'bcic_iv_2a',
    'kernel_spec': os.path.join(_KERNELS_DIR, 'bcic_iv_2a.py'),
    'subject_parser': {"type": "slice", "start": 0, "end": 3},
    'tasks': {
        'motor_imagery_left_right_foot_tongue_4class': _task(
            _EVENT_HDF5, {'tmin': 0.5, 'tmax': 2.5},
            {
                "Cue onset left (class 1)": 0,
                "Cue onset right (class 2)": 1,
                "Cue onset foot (class 3)": 2,
                "Cue onset tongue (class 4)": 3,
            },
            4, 'multiclass',
        ),
        'motor_imagery_left_right_2class': _task(
            _EVENT_HDF5, {'tmin': 0.5, 'tmax': 2.5},
            {
                "Cue onset left (class 1)": 0,
                "Cue onset right (class 2)": 1,
            },
            2, 'binary',
        ),
        'motor_imagery_hand_vs_feet_2class': _task(
            _EVENT_HDF5, {'tmin': 0.5, 'tmax': 2.5},
            {
                "Cue onset left (class 1)": 0,
                "Cue onset right (class 2)": 0,
                "Cue onset foot (class 3)": 1,
            },
            2, 'binary',
        ),
    },
})

# BCIC IV 2b - 2-class motor imagery (left / right).
register_dataset_dynamic('bcic_iv_2b', {
    'raw_path':    'bcic_iv_2b/',
    'domain_tag':  'bcic_iv_2b',
    'kernel_spec': os.path.join(_KERNELS_DIR, 'bcic_iv_2b.py'),
    'subject_parser': {"type": "slice", "start": 0, "end": 3},
    'tasks': {
        'motor_imagery_left_right_2class': _task(
            _EVENT_HDF5, {'tmin': -0.5, 'tmax': 4.0},
            {
                "Cue onset left (class 1)": 0,
                "Cue onset right (class 2)": 1,
            },
            2, 'binary',
        ),
    },
})

# Physionet EEGMMIDB - imagined movement variants.
register_dataset_dynamic('physionet_eegmmidb', {
    'raw_path':    'physionet_eegmmidb/',
    'domain_tag':  'physionet_eegmmidb',
    'kernel_spec': os.path.join(_KERNELS_DIR, 'physionet_eegmmidb.py'),
    'subject_parser': {"type": "slice", "start": 0, "end": 4},
    'tasks': {
        'motor_imagery_left_right_both_fists_feet_4class': _task(
            _EVENT_HDF5, {'tmin': 0.0, 'tmax': 4.0},
            {
                "Imagine Opening Left Fist Movement": 0,
                "Imagine Opening Right Fist Movement": 1,
                "Imagine Opening Both Fists Movement": 2,
                "Imagine Opening Feet Movement": 3,
            },
            4, 'multiclass',
        ),
        'motor_imagery_left_right_2class': _task(
            _EVENT_HDF5, {'tmin': 0.0, 'tmax': 4.0},
            {
                "Imagine Opening Left Fist Movement": 0,
                "Imagine Opening Right Fist Movement": 1,
            },
            2, 'binary',
        ),
        'motor_imagery_hand_vs_feet_2class': _task(
            _EVENT_HDF5, {'tmin': 0.0, 'tmax': 4.0},
            {
                "Imagine Opening Left Fist Movement": 0,
                "Imagine Opening Right Fist Movement": 0,
                "Imagine Opening Both Fists Movement": 0,
                "Imagine Opening Feet Movement": 1,
            },
            2, 'binary',
        ),
    },
})

# FigShare LargeMI - paradigm-aware motor imagery variants.
register_dataset_dynamic('figshare_largemi', {
    'raw_path':    'figshare_largemi/',
    'domain_tag':  'figshare_largemi',
    'kernel_spec': os.path.join(_KERNELS_DIR, 'figshare_largemi.py'),
    'subject_parser': {"type": "regex", "pattern": "(Subject[A-Z])", "group": 1},
    'tasks': {
        'motor_imagery_with_rest_11class': _task(
            _EVENT_HDF5, {'tmin': 0.0, 'tmax': 0.85},
            {
                "CLA: left hand MI": 0,
                "FREEFORM: left hand MI": 0,
                "HaLT: left hand MI": 0,
                "CLA: right hand MI": 1,
                "FREEFORM: right hand MI": 1,
                "HaLT: right hand MI": 1,
                "CLA: passive state": 2,
                "FREEFORM: passive state": 2,
                "HaLT: passive state": 2,
                "HaLT: left leg MI": 3,
                "HaLT: tongue MI": 4,
                "HaLT: right leg MI": 5,
                "5F: thumb MI": 6,
                "5F: index finger MI": 7,
                "5F: middle finger MI": 8,
                "5F: ring finger MI": 9,
                "5F: pinkie MI": 10,
            },
            11, 'multiclass',
        ),
        'motor_imagery_left_right_2class': _task(
            _EVENT_HDF5, {'tmin': 0.0, 'tmax': 3.0},
            {
                "CLA: left hand MI": 0,
                "FREEFORM: left hand MI": 0,
                "HaLT: left hand MI": 0,
                "CLA: right hand MI": 1,
                "FREEFORM: right hand MI": 1,
                "HaLT: right hand MI": 1,
            },
            2, 'binary',
        ),
        'motor_imagery_left_right_passive_3class': _task(
            _EVENT_HDF5, {'tmin': 0.0, 'tmax': 3.0},
            {
                "CLA: left hand MI": 0,
                "FREEFORM: left hand MI": 0,
                "HaLT: left hand MI": 0,
                "CLA: right hand MI": 1,
                "FREEFORM: right hand MI": 1,
                "HaLT: right hand MI": 1,
                "CLA: passive state": 2,
                "FREEFORM: passive state": 2,
                "HaLT: passive state": 2,
            },
            3, 'multiclass',
        ),
        'motor_imagery_left_right_hand_leg_4class': _task(
            _EVENT_HDF5, {'tmin': 0.0, 'tmax': 3.0},
            {
                "HaLT: left hand MI": 0,
                "HaLT: right hand MI": 1,
                "HaLT: left leg MI": 2,
                "HaLT: right leg MI": 3,
            },
            4, 'multiclass',
        ),
        'motor_imagery_hand_vs_leg_2class': _task(
            _EVENT_HDF5, {'tmin': 0.0, 'tmax': 3.0},
            {
                "HaLT: left hand MI": 0,
                "HaLT: right hand MI": 0,
                "HaLT: left leg MI": 1,
                "HaLT: right leg MI": 1,
            },
            2, 'binary',
        ),
        'motor_imagery_left_right_hand_leg_tongue_5class': _task(
            _EVENT_HDF5, {'tmin': 0.0, 'tmax': 3.0},
            {
                "HaLT: left hand MI": 0,
                "HaLT: right hand MI": 1,
                "HaLT: left leg MI": 2,
                "HaLT: right leg MI": 3,
                "HaLT: tongue MI": 4,
            },
            5, 'multiclass',
        ),
        'motor_imagery_fingers_5class': _task(
            _EVENT_HDF5, {'tmin': 0.0, 'tmax': 3.0},
            {
                "5F: thumb MI": 0,
                "5F: index finger MI": 1,
                "5F: middle finger MI": 2,
                "5F: ring finger MI": 3,
                "5F: pinkie MI": 4,
            },
            5, 'multiclass',
        ),
    },
})

# FigShare Meng2019 - cursor control variants.
register_dataset_dynamic('figshare_meng2019', {
    'raw_path':    'figshare_meng2019/',
    'domain_tag':  'figshare_meng2019',
    'kernel_spec': os.path.join(_KERNELS_DIR, 'figshare_meng2019.py'),
    'subject_parser': {"type": "split", "sep": "_", "index": 0},
    'tasks': {
        'cursor_control_minus_plus_2class': _task(
            _EVENT_HDF5, {'tmin': 1.5, 'tmax': 4.5},
            {
                "target cursor minus": 0,
                "target cursor plus": 1,
            },
            2, 'binary',
        ),
        'cursor_control_minus_plus_rest_3class': _task(
            _EVENT_HDF5, {'tmin': 1.5, 'tmax': 4.5},
            {
                "target cursor minus": 0,
                "target cursor plus": 1,
                "target rest": 2,
            },
            3, 'multiclass',
        ),
        'cursor_movement_detection_2class': _task(
            _EVENT_HDF5, {'tmin': 1.5, 'tmax': 4.5},
            {
                "target rest": 0,
                "target cursor minus": 1,
                "target cursor plus": 1,
            },
            2, 'binary',
        ),
    },
})

# FigShare SHUDB - 2-class motor imagery (left / right).
register_dataset_dynamic('figshare_shudb', {
    'raw_path':    'figshare_shudb/',
    'domain_tag':  'figshare_shudb',
    'kernel_spec': os.path.join(_KERNELS_DIR, 'figshare_shudb.py'),
    'subject_parser': {"type": "slice", "start": 0, "end": 8},
    'tasks': {
        'motor_imagery_left_right_2class': _task(
            _EVENT_HDF5, {'tmin': 0.0, 'tmax': 4.0},
            {"left": 0, "right": 1},
            2, 'binary',
        ),
    },
})

# FigShare Stroke - 2-class motor imagery for stroke patients.
register_dataset_dynamic('figshare_stroke', {
    'raw_path':    'figshare_stroke/',
    'domain_tag':  'figshare_stroke',
    'kernel_spec': os.path.join(_KERNELS_DIR, 'figshare_stroke.py'),
    'subject_parser': {"type": "slice", "start": 0, "end": 7},
    'tasks': {
        'motor_imagery_left_right_2class': _task(
            _EVENT_HDF5, {'tmin': 1.0, 'tmax': 4.0},
            {
                "Motor imagery beginning - Left hand": 0,
                "Motor imagery beginning - Right hand": 1,
            },
            2, 'binary',
        ),
    },
})

# Other High Gamma Dataset - 4-class and standard binary MI variants.
register_dataset_dynamic('other_highgammadataset', {
    'raw_path':    'other_highgammadataset/',
    'domain_tag':  'other_highgammadataset',
    'kernel_spec': os.path.join(_KERNELS_DIR, 'other_highgammadataset.py'),
    'subject_parser': {"type": "slice", "start": 0, "end": 1},
    'tasks': {
        'motor_imagery_left_right_feet_tongue_4class': _task(
            _EVENT_HDF5, {'tmin': -0.5, 'tmax': 4.0},
            {
                "left_hand": 0,
                "right_hand": 1,
                "feet": 2,
                "tongue": 3,
            },
            4, 'multiclass',
        ),
        'motor_imagery_left_right_2class': _task(
            _EVENT_HDF5, {'tmin': -0.5, 'tmax': 4.0},
            {"left_hand": 0, "right_hand": 1},
            2, 'binary',
        ),
        'motor_imagery_hand_vs_feet_2class': _task(
            _EVENT_HDF5, {'tmin': -0.5, 'tmax': 4.0},
            {
                "left_hand": 0,
                "right_hand": 0,
                "feet": 1,
            },
            2, 'binary',
        ),
    },
})

# Other OpenBMI - 2-class motor imagery (left / right hand).
_OPENBMI_MI_RECORDINGS = {
    "basename_regex": r"^sess\d{2}_subj\d{2}_EEG_MI\.mat$",
}

register_dataset_dynamic('other_openbmi', {
    'raw_path':    'other_openbmi/OpenBMI/',
    'domain_tag':  'other_openbmi',
    'kernel_spec': os.path.join(_KERNELS_DIR, 'other_openbmi.py'),
    'subject_parser': {"type": "regex", "pattern": "(subj\\d{2})", "group": 1},
    'tasks': {
        'motor_imagery_left_right_2class': _task(
            _EVENT_HDF5, {'tmin': 0.0, 'tmax': 4.0},
            {
                "left hand MI": 0,
                "right hand MI": 1,
            },
            2, 'binary',
            recording_filter=_OPENBMI_MI_RECORDINGS,
        ),
    },
})

# Other MigraineDB - stimulation modality, frequency, and diagnosis variants.
_MIGRAINE_SSVEP_RECORDINGS = {
    "basename_regex": r"(?i)^[CM]\d+(?:_\d+)?_SSVEP(?:_\d+)?\.bdf$",
}
_MIGRAINE_SSAEP_RECORDINGS = {
    "basename_regex": r"(?i)^[CM]\d+(?:_\d+)?_SSAEP(?:_\d+)?\.bdf$",
}
_MIGRAINE_RESTING_RECORDINGS = {
    "basename_regex": r"(?i)^[CM]\d+(?:_\d+)?_?Resting(?:_\d+)?\.bdf$",
}

register_dataset_dynamic('other_migrainedb', {
    'raw_path':    'other_migrainedb/',
    'domain_tag':  'other_migrainedb',
    'kernel_spec': os.path.join(_KERNELS_DIR, 'other_migrainedb.py'),
    'subject_parser': {"type": "split", "sep": "_", "index": 0},
    'tasks': {
        'ssvep_migraine_detection_2class': _task(
            _EVENT_HDF5, {'tmin': 0.0, 'tmax': 2.0},
            {
                "4Hz vision stimulation start for Health": 0,
                "4Hz vision stimulation start for migraine": 1,
                "6Hz vision stimulation start for Health": 0,
                "6Hz vision stimulation start for migraine": 1,
            },
            2, 'binary',
            recording_filter=_MIGRAINE_SSVEP_RECORDINGS,
        ),
        'ssaep_migraine_detection_2class': _task(
            _EVENT_HDF5, {'tmin': 0.0, 'tmax': 2.0},
            {
                "4Hz auditory stimulation start for Health": 0,
                "4Hz auditory stimulation start for migraine": 1,
                "6Hz auditory stimulation start for Health": 0,
                "6Hz auditory stimulation start for migraine": 1,
            },
            2, 'binary',
            recording_filter=_MIGRAINE_SSAEP_RECORDINGS,
        ),
        'resting_migraine_detection_2class': _task(
            _LONG_EVENT_HDF5, {'tmin': 0.0, 'tmax': 5.0, 'overlap': 0.5},
            {
                "resting Health": 0,
                "resting migraine": 1,
            },
            2, 'binary',
            recording_filter=_MIGRAINE_RESTING_RECORDINGS,
        ),
        'ssvep_frequency_2class': _task(
            _EVENT_HDF5, {'tmin': 0.0, 'tmax': 2.0},
            {
                "4Hz vision stimulation start for Health": 0,
                "4Hz vision stimulation start for migraine": 0,
                "6Hz vision stimulation start for Health": 1,
                "6Hz vision stimulation start for migraine": 1,
            },
            2, 'binary',
            recording_filter=_MIGRAINE_SSVEP_RECORDINGS,
        ),
        'ssaep_frequency_2class': _task(
            _EVENT_HDF5, {'tmin': 0.0, 'tmax': 2.0},
            {
                "4Hz auditory stimulation start for Health": 0,
                "4Hz auditory stimulation start for migraine": 0,
                "6Hz auditory stimulation start for Health": 1,
                "6Hz auditory stimulation start for migraine": 1,
            },
            2, 'binary',
            recording_filter=_MIGRAINE_SSAEP_RECORDINGS,
        ),
        'ssvep_4hz_migraine_detection_2class': _task(
            _EVENT_HDF5, {'tmin': 0.0, 'tmax': 2.0},
            {
                "4Hz vision stimulation start for Health": 0,
                "4Hz vision stimulation start for migraine": 1,
            },
            2, 'binary',
            recording_filter=_MIGRAINE_SSVEP_RECORDINGS,
        ),
        'ssvep_6hz_migraine_detection_2class': _task(
            _EVENT_HDF5, {'tmin': 0.0, 'tmax': 2.0},
            {
                "6Hz vision stimulation start for Health": 0,
                "6Hz vision stimulation start for migraine": 1,
            },
            2, 'binary',
            recording_filter=_MIGRAINE_SSVEP_RECORDINGS,
        ),
        'ssaep_4hz_migraine_detection_2class': _task(
            _EVENT_HDF5, {'tmin': 0.0, 'tmax': 2.0},
            {
                "4Hz auditory stimulation start for Health": 0,
                "4Hz auditory stimulation start for migraine": 1,
            },
            2, 'binary',
            recording_filter=_MIGRAINE_SSAEP_RECORDINGS,
        ),
        'ssaep_6hz_migraine_detection_2class': _task(
            _EVENT_HDF5, {'tmin': 0.0, 'tmax': 2.0},
            {
                "6Hz auditory stimulation start for Health": 0,
                "6Hz auditory stimulation start for migraine": 1,
            },
            2, 'binary',
            recording_filter=_MIGRAINE_SSAEP_RECORDINGS,
        ),
    },
})

# Physionet CAP Sleep Database - sleep staging and CAP variants.
register_dataset_dynamic('physionet_capslpdb', {
    'raw_path':    'physionet_capslpdb/',
    'domain_tag':  'physionet_capslpdb',
    'kernel_spec': os.path.join(_KERNELS_DIR, 'physionet_capslpdb.py'),
    'subject_parser': {"type": "regex", "pattern": "([A-Z]+\\d+)", "group": 1},
    'tasks': {
        'sleep_staging_30s_6class': _task(
            _LONG_EVENT_HDF5, {'tmin': 0.0, 'tmax': 30.0, 'overlap': 0.0},
            {
                "SLEEP-S0": 0,
                "SLEEP-S1": 1,
                "SLEEP-S2": 2,
                "SLEEP-S3": 3,
                "SLEEP-S4": 4,
                "SLEEP-REM": 5,
            },
            6, 'multiclass',
            recording_filter=_CAPSLPDB_REQUIRED_EEG_CHANNELS,
            dynamic_channel_system=_CAPSLPDB_DYNAMIC_CHANNEL_SYSTEM,
        ),
        'sleep_staging_4s_6class': _task(
            _LONG_EVENT_HDF5, {'tmin': 0.0, 'tmax': 4.0, 'overlap': 0.0},
            {
                "SLEEP-S0": 0,
                "SLEEP-S1": 1,
                "SLEEP-S2": 2,
                "SLEEP-S3": 3,
                "SLEEP-S4": 4,
                "SLEEP-REM": 5,
            },
            6, 'multiclass',
            recording_filter=_CAPSLPDB_REQUIRED_EEG_CHANNELS,
            dynamic_channel_system=_CAPSLPDB_DYNAMIC_CHANNEL_SYSTEM,
        ),
        'sleep_staging_30s_5class': _task(
            _LONG_EVENT_HDF5, {'tmin': 0.0, 'tmax': 30.0, 'overlap': 0.0},
            {
                "SLEEP-S0": 0,
                "SLEEP-S1": 1,
                "SLEEP-S2": 2,
                "SLEEP-S3": 3,
                "SLEEP-S4": 3,
                "SLEEP-REM": 4,
            },
            5, 'multiclass',
            recording_filter=_CAPSLPDB_REQUIRED_EEG_CHANNELS,
            dynamic_channel_system=_CAPSLPDB_DYNAMIC_CHANNEL_SYSTEM,
        ),
        'sleep_staging_4s_5class': _task(
            _LONG_EVENT_HDF5, {'tmin': 0.0, 'tmax': 4.0, 'overlap': 0.0},
            {
                "SLEEP-S0": 0,
                "SLEEP-S1": 1,
                "SLEEP-S2": 2,
                "SLEEP-S3": 3,
                "SLEEP-S4": 3,
                "SLEEP-REM": 4,
            },
            5, 'multiclass',
            recording_filter=_CAPSLPDB_REQUIRED_EEG_CHANNELS,
            dynamic_channel_system=_CAPSLPDB_DYNAMIC_CHANNEL_SYSTEM,
        ),
        'sleep_wake_detection_30s_2class': _task(
            _LONG_EVENT_HDF5, {'tmin': 0.0, 'tmax': 30.0, 'overlap': 0.0},
            {
                "SLEEP-S0": 0,
                "SLEEP-S1": 1,
                "SLEEP-S2": 1,
                "SLEEP-S3": 1,
                "SLEEP-S4": 1,
                "SLEEP-REM": 1,
            },
            2, 'binary',
            recording_filter=_CAPSLPDB_REQUIRED_EEG_CHANNELS,
            dynamic_channel_system=_CAPSLPDB_DYNAMIC_CHANNEL_SYSTEM,
        ),
        'sleep_wake_detection_4s_2class': _task(
            _LONG_EVENT_HDF5, {'tmin': 0.0, 'tmax': 4.0, 'overlap': 0.0},
            {
                "SLEEP-S0": 0,
                "SLEEP-S1": 1,
                "SLEEP-S2": 1,
                "SLEEP-S3": 1,
                "SLEEP-S4": 1,
                "SLEEP-REM": 1,
            },
            2, 'binary',
            recording_filter=_CAPSLPDB_REQUIRED_EEG_CHANNELS,
            dynamic_channel_system=_CAPSLPDB_DYNAMIC_CHANNEL_SYSTEM,
        ),
        'sleep_wake_nrem_rem_30s_3class': _task(
            _LONG_EVENT_HDF5, {'tmin': 0.0, 'tmax': 30.0, 'overlap': 0.0},
            {
                "SLEEP-S0": 0,
                "SLEEP-S1": 1,
                "SLEEP-S2": 1,
                "SLEEP-S3": 1,
                "SLEEP-S4": 1,
                "SLEEP-REM": 2,
            },
            3, 'multiclass',
            recording_filter=_CAPSLPDB_REQUIRED_EEG_CHANNELS,
            dynamic_channel_system=_CAPSLPDB_DYNAMIC_CHANNEL_SYSTEM,
        ),
        'sleep_wake_nrem_rem_4s_3class': _task(
            _LONG_EVENT_HDF5, {'tmin': 0.0, 'tmax': 4.0, 'overlap': 0.0},
            {
                "SLEEP-S0": 0,
                "SLEEP-S1": 1,
                "SLEEP-S2": 1,
                "SLEEP-S3": 1,
                "SLEEP-S4": 1,
                "SLEEP-REM": 2,
            },
            3, 'multiclass',
            recording_filter=_CAPSLPDB_REQUIRED_EEG_CHANNELS,
            dynamic_channel_system=_CAPSLPDB_DYNAMIC_CHANNEL_SYSTEM,
        ),
        'cap_detection_14s_2class': _task(
            _LONG_EVENT_HDF5, {'tmin': 0.0, 'tmax': 14.0, 'overlap': 0.0},
            {"B": 0, "A1": 1, "A2": 1, "A3": 1},
            2, 'binary',
            recording_filter=_CAPSLPDB_REQUIRED_EEG_CHANNELS,
            dynamic_channel_system=_CAPSLPDB_DYNAMIC_CHANNEL_SYSTEM,
        ),
        'cap_subtype_4s_3class': _task(
            _LONG_EVENT_HDF5, {'tmin': 0.0, 'tmax': 4.0, 'overlap': 0.0},
            {"A1": 0, "A2": 1, "A3": 2},
            3, 'multiclass',
            recording_filter=_CAPSLPDB_REQUIRED_EEG_CHANNELS,
            dynamic_channel_system=_CAPSLPDB_DYNAMIC_CHANNEL_SYSTEM,
        ),
    },
})

# Physionet CHB-MIT - seizure detection.
register_dataset_dynamic('physionet_chbmit', {
    'raw_path':    'physionet_chbmit/',
    'domain_tag':  'physionet_chbmit',
    'kernel_spec': os.path.join(_KERNELS_DIR, 'physionet_chbmit.py'),
    'subject_parser': {"type": "split", "sep": "_", "index": 0},
    'tasks': {
        'seizure_detection_2class': _task(
            _LONG_EVENT_HDF5, {'tmin': 0.0, 'tmax': 2.0, 'overlap': 0.0},
            {"interictal": 0, "seizure": 1},
            2, 'binary',
            recording_filter=_CHBMIT_REQUIRED_EEG_CHANNELS,
            dynamic_channel_system=_CHBMIT_DYNAMIC_CHANNEL_SYSTEM,
        ),
    },
})

# Physionet HMC Sleep Staging - AASM sleep staging variants.
register_dataset_dynamic('physionet_hmcsleepstaging', {
    'raw_path':    'physionet_hmcsleepstaging/',
    'domain_tag':  'physionet_hmcsleepstaging',
    'kernel_spec': os.path.join(_KERNELS_DIR, 'physionet_hmcsleepstaging.py'),
    'subject_parser': {"type": "slice", "start": 0, "end": 5},
    'tasks': {
        'sleep_staging_30s_5class': _task(
            _LONG_EVENT_HDF5, {'tmin': 0.0, 'tmax': 30.0, 'overlap': 0.0},
            {
                "Sleep stage W": 0,
                "Sleep stage N1": 1,
                "Sleep stage N2": 2,
                "Sleep stage N3": 3,
                "Sleep stage R": 4,
            },
            5, 'multiclass',
            dynamic_channel_system=_HMCSLEEP_DYNAMIC_CHANNEL_SYSTEM,
        ),
        'sleep_wake_detection_30s_2class': _task(
            _LONG_EVENT_HDF5, {'tmin': 0.0, 'tmax': 30.0, 'overlap': 0.0},
            {
                "Sleep stage W": 0,
                "Sleep stage N1": 1,
                "Sleep stage N2": 1,
                "Sleep stage N3": 1,
                "Sleep stage R": 1,
            },
            2, 'binary',
            dynamic_channel_system=_HMCSLEEP_DYNAMIC_CHANNEL_SYSTEM,
        ),
        'sleep_wake_nrem_rem_30s_3class': _task(
            _LONG_EVENT_HDF5, {'tmin': 0.0, 'tmax': 30.0, 'overlap': 0.0},
            {
                "Sleep stage W": 0,
                "Sleep stage N1": 1,
                "Sleep stage N2": 1,
                "Sleep stage N3": 1,
                "Sleep stage R": 2,
            },
            3, 'multiclass',
            dynamic_channel_system=_HMCSLEEP_DYNAMIC_CHANNEL_SYSTEM,
        ),
    },
})

# Physionet Sleep-EDF Expanded - S3/S4 merged sleep staging variants.
_SLEEPEDFX_PSG_RECORDINGS = {
    "basename_regex": r"^S[CT]\d+[A-Z]\d-PSG\.edf$",
}

register_dataset_dynamic('physionet_sleepedfx', {
    'raw_path':    'physionet_sleepedfx/',
    'domain_tag':  'physionet_sleepedfx',
    'kernel_spec': os.path.join(_KERNELS_DIR, 'physionet_sleepedfx.py'),
    'subject_parser': {"type": "regex", "pattern": "(S[CT]\\d+)", "group": 1},
    'tasks': {
        'sleep_staging_30s_5class': _task(
            _LONG_EVENT_HDF5, {'tmin': 0.0, 'tmax': 30.0, 'overlap': 0.0},
            {
                "Sleep stage W": 0,
                "Sleep stage 1": 1,
                "Sleep stage 2": 2,
                "Sleep stage 3": 3,
                "Sleep stage 4": 3,
                "Sleep stage R": 4,
            },
            5, 'multiclass',
            recording_filter=_SLEEPEDFX_PSG_RECORDINGS,
            dynamic_channel_system=_SLEEPEDFX_DYNAMIC_CHANNEL_SYSTEM,
        ),
        'sleep_wake_detection_30s_2class': _task(
            _LONG_EVENT_HDF5, {'tmin': 0.0, 'tmax': 30.0, 'overlap': 0.0},
            {
                "Sleep stage W": 0,
                "Sleep stage 1": 1,
                "Sleep stage 2": 1,
                "Sleep stage 3": 1,
                "Sleep stage 4": 1,
                "Sleep stage R": 1,
            },
            2, 'binary',
            recording_filter=_SLEEPEDFX_PSG_RECORDINGS,
            dynamic_channel_system=_SLEEPEDFX_DYNAMIC_CHANNEL_SYSTEM,
        ),
        'sleep_wake_nrem_rem_30s_3class': _task(
            _LONG_EVENT_HDF5, {'tmin': 0.0, 'tmax': 30.0, 'overlap': 0.0},
            {
                "Sleep stage W": 0,
                "Sleep stage 1": 1,
                "Sleep stage 2": 1,
                "Sleep stage 3": 1,
                "Sleep stage 4": 1,
                "Sleep stage R": 2,
            },
            3, 'multiclass',
            recording_filter=_SLEEPEDFX_PSG_RECORDINGS,
            dynamic_channel_system=_SLEEPEDFX_DYNAMIC_CHANNEL_SYSTEM,
        ),
    },
})

# TUH EEG Abnormal - normal vs abnormal EEG.
register_dataset_dynamic('tuh_eeg_abnormal', {
    'raw_path':    'tuh_eeg_abnormal/',
    'domain_tag':  'tuh_eeg_abnormal',
    'kernel_spec': os.path.join(_KERNELS_DIR, 'tuh_eeg_abnormal.py'),
    'subject_parser': {"type": "slice", "start": 0, "end": 8},
    'tasks': {
        'abnormal_eeg_detection_2class': _task(
            _LONG_EVENT_HDF5, {'tmin': 0.0, 'tmax': 5.0, 'overlap': 0.5},
            {"normal": 0, "abnormal": 1},
            2, 'binary',
        ),
    },
})

# TUH EEG Epilepsy - epilepsy vs no-epilepsy patient classification.
register_dataset_dynamic('tuh_eeg_epilepsy', {
    'raw_path':    'tuh_eeg_epilepsy/',
    'domain_tag':  'tuh_eeg_epilepsy',
    'kernel_spec': os.path.join(_KERNELS_DIR, 'tuh_eeg_epilepsy.py'),
    'subject_parser': {"type": "slice", "start": 0, "end": 8},
    'tasks': {
        'epilepsy_detection_2class': _task(
            _LONG_EVENT_HDF5, {'tmin': 0.0, 'tmax': 5.0, 'overlap': 0.5},
            {"no epilepsy": 0, "epilepsy": 1},
            2, 'binary',
        ),
    },
})

# TUH EEG Slowing - background slowing variants.
register_dataset_dynamic('tuh_eeg_slowing', {
    'raw_path':    'tuh_eeg_slowing/',
    'domain_tag':  'tuh_eeg_slowing',
    'kernel_spec': os.path.join(_KERNELS_DIR, 'tuh_eeg_slowing.py'),
    'subject_parser': {"type": "slice", "start": 0, "end": 8},
    'tasks': {
        'background_slowing_detection_2class': _task(
            _LONG_EVENT_HDF5, {'tmin': 0.0, 'tmax': 5.0, 'overlap': 0.5},
            {"bckg": 0, "slow": 1},
            2, 'binary',
        ),
        'background_slowing_seizure_3class': _task(
            _LONG_EVENT_HDF5, {'tmin': 0.0, 'tmax': 5.0, 'overlap': 0.5},
            {"bckg": 0, "slow": 1, "seiz": 2},
            3, 'multiclass',
        ),
    },
})

# TUH EEG Events - epileptiform event classification and detection.
_TUH_EVENTS_EDF_RECORDINGS = {
    "basename_regex": r".*\.edf$",
}

register_dataset_dynamic('tuh_eeg_events', {
    'raw_path':    'tuh_eeg_events/',
    'domain_tag':  'tuh_eeg_events',
    'kernel_spec': os.path.join(_KERNELS_DIR, 'tuh_eeg_events.py'),
    'subject_parser': {"type": "slice", "start": 0, "end": 8},
    'tasks': {
        'epileptiform_event_4class': _task(
            _EVENT_HDF5, {'tmin': -0.5, 'tmax': 1.0},
            {
                "background": 0,
                "spike and sharp wave": 1,
                "generalized periodic epileptiform discharges": 2,
                "periodic lateralized epileptiform discharges": 3,
            },
            4, 'multiclass',
            recording_filter=_TUH_EVENTS_EDF_RECORDINGS,
        ),
        'epileptiform_event_detection_2class': _task(
            _EVENT_HDF5, {'tmin': -0.5, 'tmax': 1.0},
            {
                "background": 0,
                "spike and sharp wave": 1,
                "generalized periodic epileptiform discharges": 1,
                "periodic lateralized epileptiform discharges": 1,
            },
            2, 'binary',
            recording_filter=_TUH_EVENTS_EDF_RECORDINGS,
        ),
    },
})

# Physionet ERP-BCI - P300 target vs non-target classification.
register_dataset_dynamic('physionet_erpbci', {
    'raw_path':    'physionet_P300/files/',
    'domain_tag':  'physionet_erpbci',
    'kernel_spec': os.path.join(_KERNELS_DIR, 'physionet_erpbci.py'),
    'subject_parser': {"type": "slice", "start": 0, "end": 4},
    'tasks': {
        'p300_target_detection_2class': _task(
            _EVENT_HDF5, {'tmin': -0.1, 'tmax': 0.8},
            {
                "unseen (row)": 0,
                "unseen (column)": 0,
                "seen (row)": 1,
                "seen (column)": 1,
            },
            2, 'binary',
            recording_filter={"basename_regex": r".*\.edf$"},
        ),
    },
})

_KAGGLE_INRIA_COMMON_EEG_CHANNELS = [
    "Fp1", "Fp2", "AF7", "AF3", "AF4", "AF8",
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "POz", "PO8", "O1", "O2",
]

_EEGMAT_COMMON_EEG_CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "F7", "F8", "T3", "T4", "C3", "C4",
    "T5", "T6", "P3", "P4", "O1", "O2", "Fz", "Cz", "Pz",
]

_UCDDB_COMMON_EEG_CHANNELS = ["C3-A2", "C4-A1"]
_UCDDB_REC_RECORDINGS = {"basename_regex": r"^ucddb\d+\.rec$"}

_TUH_ARTIFACT_LABELS = [
    "eyem", "musc", "elec", "eyem_musc", "musc_elec", "chew", "eyem_elec",
    "eyem_chew", "shiv", "chew_musc", "elpp", "chew_elec", "eyem_shiv",
    "shiv_elec",
]
_TUH_ARTIFACT_DETECTION_EVENTS = {"bckg": 0}
_TUH_ARTIFACT_DETECTION_EVENTS.update({label: 1 for label in _TUH_ARTIFACT_LABELS})

_TUH_SEIZURE_LABELS = [
    "fnsz", "gnsz", "cpsz", "absz", "spsz", "tcsz", "tnsz", "seiz", "mysz",
]
_TUH_SEIZURE_DETECTION_EVENTS = {"bckg": 0}
_TUH_SEIZURE_DETECTION_EVENTS.update({label: 1 for label in _TUH_SEIZURE_LABELS})

# Kaggle INRIA - BCI Challenge feedback correctness P300 task.
register_dataset_dynamic('kaggle_inria', {
    'raw_path':    'kaggle_inria/',
    'domain_tag':  'kaggle_inria',
    'kernel_spec': os.path.join(_KERNELS_DIR, 'kaggle_inria.py'),
    'subject_parser': {"type": "regex", "pattern": r"(S\d+)", "group": 1},
    'tasks': {
        'p300_feedback_correctness_2class': _task(
            _EVENT_HDF5, {'tmin': 0.0, 'tmax': 1.0},
            {
                "wrong feedback to P300": 0,
                "good feedback to P300": 1,
            },
            2, 'binary',
            recording_filter={"basename_regex": r"^Data_S\d+_Sess\d+\.csv$"},
            common_eeg_channels=_KAGGLE_INRIA_COMMON_EEG_CHANNELS,
        ),
    },
})

# Physionet EEG-MAT - resting/background vs mental arithmetic.
register_dataset_dynamic('physionet_eegmat', {
    'raw_path':    'physionet_eegmat/',
    'domain_tag':  'physionet_eegmat',
    'kernel_spec': os.path.join(_KERNELS_DIR, 'physionet_eegmat.py'),
    'subject_parser': {"type": "regex", "pattern": r"(Subject\d+)", "group": 1},
    'tasks': {
        'mental_arithmetic_rest_2class': _task(
            _LONG_EVENT_HDF5, {'tmin': 0.0, 'tmax': 4.0, 'overlap': 0.5},
            {
                "background_eeg": 0,
                "mental_arithmetic": 1,
            },
            2, 'binary',
            recording_filter={"basename_regex": r"^Subject\d+_[12]\.edf$"},
            common_eeg_channels=_EEGMAT_COMMON_EEG_CHANNELS,
        ),
    },
})

# Physionet UCDDB - fixed 30 s R&K sleep stage tasks on the EEG channels only.
register_dataset_dynamic('physionet_ucddb', {
    'raw_path':    'physionet_ucddb/physionet.org/files/ucddb/1.0.0/',
    'domain_tag':  'physionet_ucddb',
    'kernel_spec': os.path.join(_KERNELS_DIR, 'physionet_ucddb.py'),
    'subject_parser': {"type": "regex", "pattern": r"(ucddb\d+)", "group": 1},
    'tasks': {
        'sleep_staging_30s_6class': _task(
            _EVENT_HDF5, {'tmin': 0.0, 'tmax': 30.0},
            {
                "Sleep stage W": 0,
                "Sleep stage 1": 1,
                "Sleep stage 2": 2,
                "Sleep stage 3": 3,
                "Sleep stage 4": 4,
                "Sleep stage REM": 5,
            },
            6, 'multiclass',
            recording_filter=_UCDDB_REC_RECORDINGS,
            dynamic_channel_system=_UCDDB_DYNAMIC_CHANNEL_SYSTEM,
            common_eeg_channels=_UCDDB_COMMON_EEG_CHANNELS,
        ),
        'sleep_staging_30s_5class': _task(
            _EVENT_HDF5, {'tmin': 0.0, 'tmax': 30.0},
            {
                "Sleep stage W": 0,
                "Sleep stage 1": 1,
                "Sleep stage 2": 2,
                "Sleep stage 3": 3,
                "Sleep stage 4": 3,
                "Sleep stage REM": 4,
            },
            5, 'multiclass',
            recording_filter=_UCDDB_REC_RECORDINGS,
            dynamic_channel_system=_UCDDB_DYNAMIC_CHANNEL_SYSTEM,
            common_eeg_channels=_UCDDB_COMMON_EEG_CHANNELS,
        ),
        'sleep_wake_detection_30s_2class': _task(
            _EVENT_HDF5, {'tmin': 0.0, 'tmax': 30.0},
            {
                "Sleep stage W": 0,
                "Sleep stage 1": 1,
                "Sleep stage 2": 1,
                "Sleep stage 3": 1,
                "Sleep stage 4": 1,
                "Sleep stage REM": 1,
            },
            2, 'binary',
            recording_filter=_UCDDB_REC_RECORDINGS,
            dynamic_channel_system=_UCDDB_DYNAMIC_CHANNEL_SYSTEM,
            common_eeg_channels=_UCDDB_COMMON_EEG_CHANNELS,
        ),
        'sleep_wake_nrem_rem_30s_3class': _task(
            _EVENT_HDF5, {'tmin': 0.0, 'tmax': 30.0},
            {
                "Sleep stage W": 0,
                "Sleep stage 1": 1,
                "Sleep stage 2": 1,
                "Sleep stage 3": 1,
                "Sleep stage 4": 1,
                "Sleep stage REM": 2,
            },
            3, 'multiclass',
            recording_filter=_UCDDB_REC_RECORDINGS,
            dynamic_channel_system=_UCDDB_DYNAMIC_CHANNEL_SYSTEM,
            common_eeg_channels=_UCDDB_COMMON_EEG_CHANNELS,
        ),
    },
})

# TUH EEG Artifact - binary artifact detection and 14-class artifact type tasks.
register_dataset_dynamic('tuh_eeg_artifact', {
    'raw_path':    'tuh_eeg_artifact/',
    'domain_tag':  'tuh_eeg_artifact',
    'kernel_spec': os.path.join(_KERNELS_DIR, 'tuh_eeg_artifact.py'),
    'subject_parser': {"type": "slice", "start": 0, "end": 8},
    'tasks': {
        'artifact_detection_2class': _task(
            _LONG_EVENT_HDF5, {'tmin': 0.0, 'tmax': 1.0, 'overlap': 0.5},
            _TUH_ARTIFACT_DETECTION_EVENTS,
            2, 'binary',
            recording_filter={"basename_regex": r".*\.edf$"},
        ),
        'artifact_type_14class': _task(
            _LONG_EVENT_HDF5, {'tmin': 0.0, 'tmax': 1.0, 'overlap': 0.5},
            {label: idx for idx, label in enumerate(_TUH_ARTIFACT_LABELS)},
            14, 'multiclass',
            recording_filter={"basename_regex": r".*\.edf$"},
        ),
    },
})

# TUH EEG Seizure Corpus - seizure detection and observed seizure-type tasks.
register_dataset_dynamic('tuh_eeg_seizure', {
    'raw_path':    'tuh_eeg_seizure/',
    'domain_tag':  'tuh_eeg_seizure',
    'kernel_spec': os.path.join(_KERNELS_DIR, 'tuh_eeg_seizure.py'),
    'subject_parser': {"type": "slice", "start": 0, "end": 8},
    'tasks': {
        'seizure_detection_2class': _task(
            _LONG_EVENT_HDF5, {'tmin': 0.0, 'tmax': 2.0, 'overlap': 0.5},
            _TUH_SEIZURE_DETECTION_EVENTS,
            2, 'binary',
            recording_filter={"basename_regex": r".*\.edf$"},
        ),
        'seizure_type_9class': _task(
            _LONG_EVENT_HDF5, {'tmin': 0.0, 'tmax': 2.0, 'overlap': 0.5},
            {label: idx for idx, label in enumerate(_TUH_SEIZURE_LABELS)},
            9, 'multiclass',
            recording_filter={"basename_regex": r".*\.edf$"},
        ),
    },
})

_SEED_COMMON_EEG_CHANNELS = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4',
    'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
    'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3',
    'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2',
    'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1',
    'O1', 'OZ', 'O2', 'CB2',
]

_FACED_COMMON_EEG_CHANNELS = [
    'Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8', 'FC1', 'FC2', 'FC5',
    'FC6', 'Cz', 'C3', 'C4', 'T7', 'T8', 'CP1', 'CP2', 'CP5', 'CP6',
    'Pz', 'P3', 'P4', 'P7', 'P8', 'PO3', 'PO4', 'Oz', 'O1', 'O2',
]

_KUL_COMMON_CHANNELS = [
    'Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7',
    'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7',
    'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9',
    'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz',
    'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4',
    'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz',
    'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2',
    'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2',
]

# Zenodo KUL - attended-side auditory attention decoding.
register_dataset_dynamic('zenodo_kul', {
    'raw_path':    'zenodo_kul/',
    'domain_tag':  'zenodo_kul',
    'kernel_spec': os.path.join(_KERNELS_DIR, 'zenodo_kul.py'),
    'subject_parser': {"type": "regex", "pattern": r"(S\d+)", "group": 1},
    'tasks': {
        'auditory_attention_attended_side_2class': _task(
            _LONG_EVENT_HDF5, {'tmin': 0.0, 'tmax': 5.0, 'overlap': 0.5},
            {
                'attended_left': 0,
                'attended_right': 1,
            },
            2, 'binary',
            recording_filter={"basename_regex": r"^S\d+\.mat$"},
            common_eeg_channels=_KUL_COMMON_CHANNELS,
        ),
    },
})

# Zenodo DTU - male/female, left/right, and combined auditory attention tasks.
register_dataset_dynamic('zenodo_dtu', {
    'raw_path':    'zenodo_dtu/Processed_EEG/',
    'domain_tag':  'zenodo_dtu',
    'kernel_spec': os.path.join(_KERNELS_DIR, 'zenodo_dtu.py'),
    'subject_parser': {"type": "regex", "pattern": r"(S\d+)", "group": 1},
    'tasks': {
        'auditory_attention_male_female_2class': _task(
            _LONG_EVENT_HDF5, {'tmin': 0.0, 'tmax': 5.0, 'overlap': 0.5},
            {
                'male_left': 0,
                'male_right': 0,
                'female_left': 1,
                'female_right': 1,
            },
            2, 'binary',
            recording_filter={"basename_regex": r"^S\d+_struct\.mat$"},
        ),
        'auditory_attention_left_right_2class': _task(
            _LONG_EVENT_HDF5, {'tmin': 0.0, 'tmax': 5.0, 'overlap': 0.5},
            {
                'male_left': 0,
                'female_left': 0,
                'male_right': 1,
                'female_right': 1,
            },
            2, 'binary',
            recording_filter={"basename_regex": r"^S\d+_struct\.mat$"},
        ),
        'auditory_attention_male_female_left_right_4class': _task(
            _LONG_EVENT_HDF5, {'tmin': 0.0, 'tmax': 5.0, 'overlap': 0.5},
            {
                'male_left': 0,
                'male_right': 1,
                'female_left': 2,
                'female_right': 3,
            },
            4, 'multiclass',
            recording_filter={"basename_regex": r"^S\d+_struct\.mat$"},
        ),
    },
})

# SEED - fixed 4 s emotion segments with preset cross-trial split metadata.
register_dataset_dynamic('other_seed', {
    'raw_path':    'other_seed/SEED_EEG/Preprocessed_EEG/',
    'domain_tag':  'other_seed',
    'kernel_spec': os.path.join(_KERNELS_DIR, 'other_seed.py'),
    'subject_parser': {"type": "split", "sep": "_", "index": 0},
    'tasks': {
        'emotion_recognition_3class': _task(
            _EVENT_HDF5, {'tmin': 0.0, 'tmax': 4.0},
            {
                'negative': 0,
                'neutral': 1,
                'positive': 2,
            },
            3, 'multiclass',
            recording_filter={"basename_regex": r"^\d+_\d{8}\.mat$"},
            common_eeg_channels=_SEED_COMMON_EEG_CHANNELS,
        ),
    },
})

# FACED - 9-class fixed 10 s emotion segments from 30 common channels.
register_dataset_dynamic('other_faced', {
    'raw_path':    'FACED/Processed_data/',
    'domain_tag':  'other_faced',
    'kernel_spec': os.path.join(_KERNELS_DIR, 'other_faced.py'),
    'subject_parser': {"type": "regex", "pattern": r"(sub\d+)", "group": 1},
    'tasks': {
        'emotion_recognition_9class': _task(
            _EVENT_HDF5, {'tmin': 0.0, 'tmax': 10.0},
            {f'emotion_class_{idx}': idx for idx in range(9)},
            9, 'multiclass',
            recording_filter={"basename_regex": r"^sub\d+\.pkl$"},
            common_eeg_channels=_FACED_COMMON_EEG_CHANNELS,
        ),
    },
})


__all__ = [
    'DATASET_REGISTRY',
    'register_dataset_dynamic',
    'get_dataset_config',
    'get_task_config',
    'list_datasets',
    'list_tasks',
    'list_all',
]
