"""Kernel for the Zenodo DTU auditory attention dataset."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import mne
import numpy as np
import scipy.io as scio

from eegunity.utils.annotation_metadata import encode_annotation_description


_SFREQ = 512.0
_MALE_FEMALE_LABELS = {1: 'male', 2: 'female'}
_LEFT_RIGHT_LABELS = {1: 'left', 2: 'right'}


def _is_struct_mat(file_path: str) -> bool:
    return str(file_path).lower().endswith('_struct.mat')


def _channel_types(channel_names):
    types = []
    for name in channel_names:
        if name == 'Status':
            types.append('stim')
        elif name.startswith('EXG'):
            types.append('misc')
        else:
            types.append('eeg')
    return types


def _description_payload(original_description: str = '') -> str:
    description_dict = {
        'original_description': original_description,
        'eegunity_description': {
            'amplifier': 'unknown',
            'cap': '64-channel EEG cap with EXG references',
            'age': 'unknown',
            'sex': 'unknown',
            'handedness': 'unknown',
        },
    }
    return json.dumps(description_dict)


def _resolve_audio_path(audio_root: str, filename) -> str:
    text = str(filename).strip()
    if not text or text == '[]':
        return ''
    return os.path.join(audio_root, text)


def _load_struct_raw(file_path: str, original_description: str = '') -> mne.io.BaseRaw | None:
    if not _is_struct_mat(file_path):
        return None

    mat = scio.loadmat(file_path, struct_as_record=False, squeeze_me=True)
    data = mat.get('data')
    expinfo = mat.get('expinfo')
    if data is None or expinfo is None:
        return None

    channel_names = list(np.atleast_1d(data.dim.chan.eeg).tolist())
    eeg = np.asarray(data.eeg, dtype=np.float64)
    if eeg.ndim != 2:
        return None

    sfreq = float(getattr(getattr(data, 'fsample', None), 'eeg', _SFREQ))
    info = mne.create_info(
        ch_names=channel_names,
        sfreq=sfreq,
        ch_types=_channel_types(channel_names),
    )
    info['description'] = _description_payload(original_description)
    rebuilt_raw = mne.io.RawArray(eeg.T, info, verbose='ERROR')

    trigger_samples = np.asarray(data.event.eeg.sample).astype(int).reshape(-1)
    trigger_values = np.asarray(data.event.eeg.value).astype(int).reshape(-1)
    audio_root = os.path.join(os.path.dirname(os.path.dirname(file_path)), 'Audio')

    onsets = []
    durations = []
    descriptions = []
    search_start = 0

    for trial_id, trial in enumerate(np.atleast_1d(expinfo), start=1):
        attend_mf = int(trial.attend_mf)
        attend_lr = int(trial.attend_lr)
        trial_trigger = int(trial.trigger)

        relative_hits = np.where(trigger_values[search_start:] == trial_trigger)[0]
        if len(relative_hits) == 0:
            continue
        start_idx = int(relative_hits[0]) + search_start
        stop_idx = start_idx + 1
        while stop_idx < len(trigger_values) and int(trigger_values[stop_idx]) != 191:
            stop_idx += 1
        if stop_idx >= len(trigger_samples):
            continue

        start_sample = int(trigger_samples[start_idx])
        stop_sample = int(trigger_samples[stop_idx])
        duration_s = (stop_sample - start_sample) / rebuilt_raw.info['sfreq']

        mf_label = _MALE_FEMALE_LABELS.get(attend_mf)
        lr_label = _LEFT_RIGHT_LABELS.get(attend_lr)
        if mf_label is None or lr_label is None:
            search_start = stop_idx + 1
            continue

        descriptions.append(encode_annotation_description(
            f'{mf_label}_{lr_label}',
            trial_id=trial_id,
            attend_mf=mf_label,
            attend_lr=lr_label,
            acoustic_condition=int(trial.acoustic_condition),
            n_speakers=int(trial.n_speakers),
            male_audio=_resolve_audio_path(audio_root, trial.wavfile_male),
            female_audio=_resolve_audio_path(audio_root, trial.wavfile_female),
        ))
        onsets.append(start_sample / rebuilt_raw.info['sfreq'])
        durations.append(duration_s)
        search_start = stop_idx + 1

    if onsets:
        rebuilt_raw.set_annotations(mne.Annotations(
            onset=onsets,
            duration=durations,
            description=descriptions,
        ))
    return rebuilt_raw


@dataclass
class ZenodoDTUKernel:
    KERNEL_ID: str = 'zenodo-dtu-v2'
    HANDLES_UNAVAILABLE: bool = True

    def load(self, row) -> mne.io.BaseRaw | None:
        file_path = str(row['File Path'])
        return _load_struct_raw(file_path)

    def apply(self, udataset, raw: mne.io.BaseRaw, row) -> mne.io.BaseRaw:
        file_path = str(row['File Path'])
        if not _is_struct_mat(file_path):
            return raw

        try:
            if raw.info.get('description') and raw.annotations is not None and len(raw.annotations) > 0:
                return raw
            rebuilt_raw = _load_struct_raw(file_path, raw.info.get('description', ''))
        except Exception:
            return raw
        return rebuilt_raw if rebuilt_raw is not None else raw


KERNEL = ZenodoDTUKernel()
