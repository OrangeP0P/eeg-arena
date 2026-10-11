"""Kernel for the Zenodo KUL auditory attention dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass

import mne
import numpy as np
import scipy.io as scio

from eegunity.utils.annotation_metadata import encode_annotation_description


_SFREQ = 128.0
_N_CHANNELS = 64
_BIOSEMI64_CHANNEL_NAMES = [
    'Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7',
    'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7',
    'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9',
    'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz',
    'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4',
    'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz',
    'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2',
    'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2',
]

_LEFT_TOKENS = {'l', 'left', '1', 'attended_left'}
_RIGHT_TOKENS = {'r', 'right', '2', 'attended_right'}


def _normalize_attended_ear(value) -> str | None:
    if value is None:
        return None

    text = str(np.asarray(value).flat[0]).strip().lower()
    if text in _LEFT_TOKENS:
        return 'attended_left'
    if text in _RIGHT_TOKENS:
        return 'attended_right'
    return None


def _channel_labels_from_trial(trial) -> list[str] | None:
    try:
        channels = trial.get('FileHeader', {}).get('Channels')
    except Exception:
        return None

    if channels is None:
        return None

    labels = []
    for channel in list(np.asarray(channels, dtype=object).flatten()):
        if isinstance(channel, dict):
            label = channel.get('Label')
        else:
            label = getattr(channel, 'Label', None)
        label = str(label).strip() if label is not None else ''
        if label:
            labels.append(label)

    if len(labels) >= _N_CHANNELS:
        return labels[:_N_CHANNELS]
    return None


def _resolve_channel_names(trials) -> list[str]:
    for trial in list(np.asarray(trials, dtype=object).flatten()):
        labels = _channel_labels_from_trial(trial)
        if labels is not None:
            return labels
    return list(_BIOSEMI64_CHANNEL_NAMES)


def _description_payload(raw: mne.io.BaseRaw) -> str:
    description_dict = {
        'original_description': raw.info.get('description', ''),
        'eegunity_description': {
            'amplifier': 'unknown',
            'cap': '64-channel EEG cap',
            'age': 'unknown',
            'sex': 'unknown',
            'handedness': 'unknown',
        },
    }
    return json.dumps(description_dict)


def _load_mat_raw(file_path: str) -> mne.io.BaseRaw | None:
    """Load zenodo_kul .mat file and return MNE Raw object."""
    if not file_path.lower().endswith('.mat'):
        return None

    try:
        mat = scio.loadmat(file_path, simplify_cells=True)
    except Exception:
        return None

    trials = mat.get('trials')
    if not trials:
        return None

    eeg_blocks = []
    onsets = []
    durations = []
    descriptions = []
    sample_offset = 0

    for trial_index, trial in enumerate(trials, start=1):
        try:
            eeg = np.asarray(trial['RawData']['EegData'], dtype=np.float64)
        except Exception:
            continue

        if eeg.ndim != 2:
            continue
        if eeg.shape[1] != _N_CHANNELS and eeg.shape[0] == _N_CHANNELS:
            eeg = eeg.T
        if eeg.shape[1] != _N_CHANNELS:
            continue

        label = _normalize_attended_ear(trial.get('attended_ear'))
        if label is None:
            continue

        stimuli = list(np.asarray(trial.get('stimuli', []), dtype=object).flatten())
        n_samples = eeg.shape[0]
        duration_s = n_samples / _SFREQ
        onset_s = sample_offset / _SFREQ

        descriptions.append(encode_annotation_description(
            label,
            trial_id=trial_index,
            attended_track=trial.get('attended_track'),
            condition=trial.get('condition'),
            experiment=trial.get('experiment'),
            part=trial.get('part'),
            repetition=trial.get('repetition'),
            stimulus_1=stimuli[0] if len(stimuli) > 0 else '',
            stimulus_2=stimuli[1] if len(stimuli) > 1 else '',
        ))
        onsets.append(onset_s)
        durations.append(duration_s)
        eeg_blocks.append(eeg)
        sample_offset += n_samples

    if not eeg_blocks:
        return None

    channel_names = _resolve_channel_names(trials)
    info = mne.create_info(channel_names, sfreq=_SFREQ, ch_types=['eeg'] * _N_CHANNELS)

    eeg_concat = np.concatenate(eeg_blocks, axis=0).T
    rebuilt_raw = mne.io.RawArray(eeg_concat, info, verbose='ERROR')
    try:
        rebuilt_raw.set_montage('biosemi64', on_missing='ignore', verbose='ERROR')
    except Exception:
        pass
    rebuilt_raw.set_annotations(mne.Annotations(
        onset=onsets,
        duration=durations,
        description=descriptions,
    ))
    return rebuilt_raw


def _description_payload_from_info(info: dict) -> str:
    description_dict = {
        'original_description': info.get('description', ''),
        'eegunity_description': {
            'amplifier': 'unknown',
            'cap': '64-channel EEG cap',
            'age': 'unknown',
            'sex': 'unknown',
            'handedness': 'unknown',
        },
    }
    return json.dumps(description_dict)


@dataclass
class ZenodoKulKernel:
    KERNEL_ID: str = 'zenodo-kul-v2'
    HANDLES_UNAVAILABLE: bool = True

    def load(self, row) -> mne.io.BaseRaw | None:
        file_path = str(row['File Path'])
        return _load_mat_raw(file_path)

    def apply(self, udataset, raw: mne.io.BaseRaw, row) -> mne.io.BaseRaw:
        file_path = str(row['File Path'])
        if not file_path.lower().endswith('.mat'):
            return raw

        # If raw already has annotations from load(), use it
        if raw.annotations is not None and len(raw.annotations) > 0:
            return raw

        # Otherwise load from file
        try:
            rebuilt_raw = _load_mat_raw(file_path)
        except Exception:
            return raw

        if rebuilt_raw is None:
            return raw

        # Set description if available from original raw
        if raw.info.get('description'):
            rebuilt_raw.info['description'] = _description_payload_from_info(raw.info)
        return rebuilt_raw


KERNEL = ZenodoKulKernel()
