"""Physionet EEG-MAT stage1 kernel for EEGUnity."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import mne
import pandas as pd

from eegunity.utils.annotation_metadata import encode_annotation_description


def _normalize_channel_names(raw: mne.io.BaseRaw) -> None:
    rename_map = {}
    for channel in raw.ch_names:
        normalized = channel
        if normalized.startswith("EEG "):
            normalized = normalized[4:]
        elif normalized == "ECG ECG":
            normalized = "ECG"
        if normalized != channel:
            rename_map[channel] = normalized

    if rename_map:
        raw.rename_channels(rename_map)

    if "ECG" in raw.ch_names:
        raw.set_channel_types({"ECG": "ecg"})


def _normalize_gender(value) -> str:
    text = str(value).strip().lower()
    if text in {"f", "female"}:
        return "female"
    if text in {"m", "male"}:
        return "male"
    return "unknown"


@dataclass
class PhysionetEegmatKernel:
    """Stage1 kernel for the physionet-eegmat dataset."""

    KERNEL_ID: str = "physionet-eegmat-stage1"

    def apply(self, udataset, raw: mne.io.BaseRaw, row) -> mne.io.BaseRaw:
        file_path = row["File Path"]
        _normalize_channel_names(raw)

        # Locate subject-info.csv.
        data_dir = os.path.dirname(file_path)
        subject_info_path = os.path.join(data_dir, "subject-info.csv")

        filename = os.path.basename(file_path)
        subject_id = filename.split("_")[0]

        age = "unknown"
        gender = "unknown"
        count_quality_description = "unknown"

        if os.path.exists(subject_info_path):
            subject_info = pd.read_csv(subject_info_path)
            subject_data = subject_info[subject_info["Subject"] == subject_id]

            if not subject_data.empty:
                age = subject_data["Age"].values[0] if "Age" in subject_data.columns else "unknown"
                gender = _normalize_gender(
                    subject_data["Gender"].values[0] if "Gender" in subject_data.columns else "unknown"
                )
                count_quality = (
                    subject_data["Count quality"].values[0]
                    if "Count quality" in subject_data.columns
                    else "unknown"
                )
                if count_quality == 1:
                    count_quality_description = "good"
                elif count_quality == 0:
                    count_quality_description = "bad"

        # Determine task from filename suffix.
        if filename.endswith("_1.edf"):
            task_description = "background_eeg"
        elif filename.endswith("_2.edf"):
            task_description = "mental_arithmetic"
        else:
            task_description = "unknown_task"

        annotation_description = encode_annotation_description(
            task_description,
            count_quality=count_quality_description,
            subject_id=subject_id,
        )

        description_dict = {
            "original_description": raw.info.get("description", ""),
            "eegunity_description": {
                "amplifier": "Neurocom EEG 23-channel system  (Ukraine, XAI-MEDICA)",
                "cap": "silver/silver chloride electrodes",
                "age": str(age),
                "sex": gender,
                "handedness": "unknown",
            },
        }
        raw.info["description"] = json.dumps(description_dict)

        # Set annotation spanning entire recording.
        annotations = mne.Annotations(
            onset=0,
            duration=raw.times[-1],
            description=annotation_description,
        )
        raw.set_annotations(annotations)

        return raw


KERNEL = PhysionetEegmatKernel()
