"""Kernel for the FACED emotion-recognition dataset."""

from __future__ import annotations

import json
import os
import pickle
import re
from dataclasses import dataclass

import mne
import numpy as np
import scipy.signal

from eegunity.utils.annotation_metadata import encode_annotation_description


_SFREQ = 200.0
_VIDEO_SECONDS = 30.0
_SEGMENT_SECONDS = 10.0
_RESAMPLED_POINTS = 6000
_SEGMENT_POINTS = int(_SFREQ * _SEGMENT_SECONDS)
_FILENAME_RE = re.compile(r'^sub(?P<subject>\d{3})\.pkl$')
_LABELS = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8])
_GROUP1_INDICES = np.concatenate([np.arange(0, 16), np.arange(18, 32)])
_GROUP2_INDICES = np.arange(0, 30)
_CHANNEL_NAMES = [
    'Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8', 'FC1', 'FC2', 'FC5',
    'FC6', 'Cz', 'C3', 'C4', 'T7', 'T8', 'CP1', 'CP2', 'CP5', 'CP6',
    'Pz', 'P3', 'P4', 'P7', 'P8', 'PO3', 'PO4', 'Oz', 'O1', 'O2',
]


def _build_video_preset_split() -> dict[int, str]:
    """Stratified video split for FACED cross_trial.

    FACED has three videos for most classes and four neutral videos. Keeping one
    video per class in val/test avoids class-missing evaluation splits.
    """
    split_by_video = {}
    for label_id in np.unique(_LABELS):
        video_ids = (np.where(_LABELS == label_id)[0] + 1).tolist()
        if len(video_ids) == 4:
            split_names = ['train', 'train', 'val', 'test']
        elif len(video_ids) == 3:
            split_names = ['train', 'val', 'test']
        else:
            raise ValueError(f'Unexpected FACED video count for label {label_id}: {len(video_ids)}')
        split_by_video.update(dict(zip(video_ids, split_names)))
    return split_by_video


_VIDEO_PRESET_SPLIT = _build_video_preset_split()


def _description_payload() -> str:
    return json.dumps({
        'original_description': '',
        'eegunity_description': {
            'amplifier': 'unknown',
            'cap': '32-channel EEG cap mapped to 30 common channels',
            'age': 'unknown',
            'sex': 'unknown',
            'handedness': 'unknown',
        },
    })


@dataclass
class OtherFacedKernel:
    KERNEL_ID: str = 'other-faced-v1'
    HANDLES_UNAVAILABLE: bool = True

    def load(self, row) -> mne.io.BaseRaw | None:
        file_path = str(row['File Path'])
        match = _FILENAME_RE.match(os.path.basename(file_path))
        if not match:
            return None

        with open(file_path, 'rb') as handle:
            raw_array = pickle.load(handle)

        eeg = np.asarray(raw_array, dtype=np.float64)
        if eeg.shape[0] != len(_LABELS):
            raise ValueError(f'Unexpected FACED video count in {file_path}: {eeg.shape}')

        subject_index = int(match.group('subject'))
        selected_indices = _GROUP1_INDICES if subject_index < 60 else _GROUP2_INDICES
        eeg = scipy.signal.resample(eeg, _RESAMPLED_POINTS, axis=2)
        eeg = eeg[:, selected_indices, :]

        blocks = []
        onsets = []
        durations = []
        descriptions = []
        sample_offset = 0

        for video_id, (video_data, label_id) in enumerate(zip(eeg, _LABELS), start=1):
            blocks.append(video_data.T)
            for segment_id in range(1, 4):
                onset_sample = sample_offset + (segment_id - 1) * _SEGMENT_POINTS
                onsets.append(onset_sample / _SFREQ)
                durations.append(_SEGMENT_SECONDS)
                descriptions.append(encode_annotation_description(
                    f'emotion_class_{int(label_id)}',
                    subject_id=f'sub{subject_index:03d}',
                    video_id=video_id,
                    segment_id=segment_id,
                    preset_split=_VIDEO_PRESET_SPLIT[video_id],
                ))
            sample_offset += _RESAMPLED_POINTS

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


KERNEL = OtherFacedKernel()
