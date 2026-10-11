"""Kernel for the SEED emotion-recognition dataset."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache

import mne
import numpy as np
import scipy.io as scio

from eegunity.utils.annotation_metadata import encode_annotation_description


_SFREQ = 200.0
_SEGMENT_SECONDS = 4.0
_SAMPLES_PER_SEGMENT = int(_SFREQ * _SEGMENT_SECONDS)
_FILENAME_RE = re.compile(r'^(?P<subject>\d+)_(?P<date>\d{8})\.mat$')
_LABEL_MAP = {-1: 'negative', 0: 'neutral', 1: 'positive'}
_CHANNEL_NAMES = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4',
    'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
    'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3',
    'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2',
    'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1',
    'O1', 'OZ', 'O2', 'CB2',
]


def _description_payload() -> str:
    return json.dumps({
        'original_description': '',
        'eegunity_description': {
            'amplifier': 'ESI NeuroScan',
            'cap': '62-channel EEG cap',
            'age': 'unknown',
            'sex': 'unknown',
            'handedness': 'unknown',
        },
    })


def _trial_preset_split(trial_id: int) -> str:
    if 1 <= trial_id <= 9:
        return 'train'
    if 10 <= trial_id <= 12:
        return 'val'
    return 'test'


def _extract_trial_keys(mat) -> list[str]:
    keys = [key for key in mat.keys() if re.search(r'eeg\d+$', key)]
    return sorted(keys, key=lambda name: int(name.split('eeg')[-1]))


@lru_cache(maxsize=1)
def _seed_labels(dataset_root: str):
    label_path = os.path.join(dataset_root, 'label.mat')
    labels = scio.loadmat(label_path)['label'].reshape(-1).tolist()
    return [int(label) for label in labels]


@lru_cache(maxsize=64)
def _subject_session_order(dataset_root: str, subject_id: str):
    dates = []
    for file_name in os.listdir(dataset_root):
        match = _FILENAME_RE.match(file_name)
        if match and match.group('subject') == subject_id:
            dates.append(match.group('date'))
    return sorted(dates)


def _parse_subject_session(file_path: str):
    match = _FILENAME_RE.match(os.path.basename(file_path))
    if not match:
        raise ValueError(f'Unsupported SEED file name: {file_path}')
    subject_id = match.group('subject')
    session_date = match.group('date')
    session_order = _subject_session_order(os.path.dirname(file_path), subject_id)
    return subject_id, session_date, session_order.index(session_date) + 1


@dataclass
class OtherSeedKernel:
    KERNEL_ID: str = 'other-seed-v1'
    HANDLES_UNAVAILABLE: bool = True

    def load(self, row) -> mne.io.BaseRaw | None:
        file_path = str(row['File Path'])
        if not _FILENAME_RE.match(os.path.basename(file_path)):
            return None

        mat = scio.loadmat(file_path)
        trial_keys = _extract_trial_keys(mat)
        labels = _seed_labels(os.path.dirname(file_path))
        if len(trial_keys) != len(labels):
            raise ValueError(
                f'SEED trial count mismatch for {file_path}: {len(trial_keys)} trials vs {len(labels)} labels'
            )

        subject_id, session_date, session_id = _parse_subject_session(file_path)

        blocks = []
        onsets = []
        durations = []
        descriptions = []
        sample_offset = 0

        for trial_id, (trial_key, raw_label) in enumerate(zip(trial_keys, labels), start=1):
            eeg = np.asarray(mat[trial_key], dtype=np.float64)
            if eeg.ndim != 2:
                continue
            if eeg.shape[0] != len(_CHANNEL_NAMES) and eeg.shape[1] == len(_CHANNEL_NAMES):
                eeg = eeg.T
            if eeg.shape[0] != len(_CHANNEL_NAMES):
                raise ValueError(f'Unexpected SEED channel shape for {file_path}:{trial_key}: {eeg.shape}')

            n_samples = eeg.shape[1]
            n_segments = n_samples // _SAMPLES_PER_SEGMENT
            base_label = _LABEL_MAP[int(raw_label)]
            preset_split = _trial_preset_split(trial_id)

            for segment_id in range(1, n_segments + 1):
                onset_sample = sample_offset + (segment_id - 1) * _SAMPLES_PER_SEGMENT
                onsets.append(onset_sample / _SFREQ)
                durations.append(_SEGMENT_SECONDS)
                descriptions.append(encode_annotation_description(
                    base_label,
                    subject_id=subject_id,
                    session_id=session_id,
                    session_date=session_date,
                    trial_id=trial_id,
                    segment_id=segment_id,
                    preset_split=preset_split,
                ))

            blocks.append(eeg.T)
            sample_offset += n_samples

        if not blocks:
            return None

        info = mne.create_info(_CHANNEL_NAMES, sfreq=_SFREQ, ch_types=['eeg'] * len(_CHANNEL_NAMES))
        info['description'] = _description_payload()
        continuous = np.concatenate(blocks, axis=0).T
        raw = mne.io.RawArray(continuous, info, verbose='ERROR')
        raw.set_annotations(mne.Annotations(
            onset=onsets,
            duration=durations,
            description=descriptions,
        ))
        return raw

    def apply(self, udataset, raw: mne.io.BaseRaw, row) -> mne.io.BaseRaw:
        return raw


KERNEL = OtherSeedKernel()
