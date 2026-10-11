"""Physionet UCD Sleep Apnea Database stage1 kernel for EEGUnity.

Sleep stage encoding in _stage.txt (R&K system, one integer per 30-second epoch):
  0 → Sleep stage W   (Wake)
  1 → Sleep stage 1   (NREM Stage 1)
  2 → Sleep stage 2   (NREM Stage 2)
  3 → Sleep stage 3   (NREM Stage 3, slow-wave sleep)
  4 → Sleep stage 4   (NREM Stage 4, deep SWS)
  5 → Sleep stage REM (REM sleep)
  8 → Sleep stage 8   (undefined / movement artifact epoch)

_lifecard.edf files (ECG-only) are skipped — no annotations applied.
Subject demographics are available in SubjectDetails.xls but not parsed here
(individual subject-to-file mapping is not in the public EDF headers).

Source: https://physionet.org/content/ucddb/1.0.0/
Kernel added: 2026-04-09 — LLM-assisted, requires human review.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import mne

from eegunity.utils.annotation_metadata import encode_annotation_description


_STAGE_MAP = {
    0: "Sleep stage W",
    1: "Sleep stage 1",
    2: "Sleep stage 2",
    3: "Sleep stage 3",
    4: "Sleep stage 4",
    5: "Sleep stage REM",
    8: "Sleep stage 8",
}

_EPOCH_DURATION = 30.0  # seconds per epoch (fixed by dataset)


def _canonical_channel_name(channel_name: str) -> str:
    """Map typed or legacy UCDDB names onto stable bare-channel labels."""
    raw_name = str(channel_name).strip()
    if ":" in raw_name:
        prefix, suffix = raw_name.split(":", 1)
        raw_name = suffix.strip() or prefix.strip()

    compact = raw_name.replace(" ", "").replace("_", "").casefold()
    canonical_map = {
        "c3a2": "C3-A2",
        "c4a1": "C4-A1",
        "lefteye": "Lefteye",
        "righteye": "RightEye",
        "emg": "EMG",
        "ecg": "ECG",
    }
    return canonical_map.get(compact, raw_name)


def _normalize_channels(raw: mne.io.BaseRaw) -> None:
    rename_map = {}
    occupied_names = set(raw.ch_names)
    for channel_name in raw.ch_names:
        canonical_name = _canonical_channel_name(channel_name)
        if canonical_name == channel_name:
            continue
        if canonical_name in occupied_names or canonical_name in rename_map.values():
            continue
        rename_map[channel_name] = canonical_name

    if rename_map:
        raw.rename_channels(rename_map)

    channel_types = {}
    for channel in raw.ch_names:
        canonical_name = _canonical_channel_name(channel)
        if canonical_name in {"C3-A2", "C4-A1"}:
            channel_types[channel] = "eeg"
        elif canonical_name in {"Lefteye", "RightEye"}:
            channel_types[channel] = "eog"
        elif canonical_name == "EMG":
            channel_types[channel] = "emg"
        elif canonical_name == "ECG":
            channel_types[channel] = "ecg"
        else:
            channel_types[channel] = "misc"

    try:
        raw.set_channel_types(channel_types, on_unit_change="ignore")
    except TypeError:
        raw.set_channel_types(channel_types)


@dataclass
class PhysionetUcddbKernel:
    """Stage1 kernel for the physionet-ucddb dataset."""

    KERNEL_ID: str = "physionet-ucddb-stage1"

    def apply(self, udataset, raw: mne.io.BaseRaw, row) -> mne.io.BaseRaw:
        file_path = row["File Path"]
        file_name = os.path.basename(file_path)
        _normalize_channels(raw)

        description_dict = {
            "original_description": raw.info.get("description", ""),
            "eegunity_description": {
                "amplifier": "unknown",
                "cap":       "unknown",
                "age":       "unknown",
                "sex":       "unknown",
                "handedness": "unknown",
            },
        }
        raw.info["description"] = json.dumps(description_dict)

        # _lifecard.edf files are ECG-only; skip annotation injection.
        if "_lifecard" in file_name:
            raw.set_annotations(mne.Annotations([], [], []))
            return raw

        # Companion stage file: ucddbXXX.rec → ucddbXXX_stage.txt
        stage_path = file_path.replace(".rec", "_stage.txt")
        if not os.path.exists(stage_path):
            return raw

        # Parse stage file — one integer per line, each = 30-second epoch.
        onsets, durations, descriptions = [], [], []
        with open(stage_path, "r") as fh:
            for epoch_idx, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                try:
                    code = int(line)
                except ValueError:
                    continue
                label = _STAGE_MAP.get(code)
                if label is None:
                    continue
                onsets.append(epoch_idx * _EPOCH_DURATION)
                durations.append(_EPOCH_DURATION)
                descriptions.append(
                    encode_annotation_description(
                        label,
                        epoch_index=epoch_idx,
                        sleep_stage_code=code,
                    )
                )

        raw.set_annotations(
            mne.Annotations(
                onset=onsets,
                duration=durations,
                description=descriptions,
            )
        )
        return raw


KERNEL = PhysionetUcddbKernel()
