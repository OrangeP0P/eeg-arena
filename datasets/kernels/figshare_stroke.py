"""Figshare Stroke stage1 kernel for EEGUnity."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import mne
import numpy as np
import pandas as pd


_SUBJECT_DICT: Dict[str, Dict[str, Any]] = {
    "01": {"sex": "Male", "Age": 45, "Handedness": "Right"},
    "02": {"sex": "Male", "Age": 60, "Handedness": "Right"},
    "03": {"sex": "Male", "Age": 60, "Handedness": "Right"},
    "04": {"sex": "Male", "Age": 56, "Handedness": "Right"},
    "05": {"sex": "Female", "Age": 44, "Handedness": "Right"},
    "06": {"sex": "Male", "Age": 66, "Handedness": "left"},
    "07": {"sex": "Male", "Age": 62, "Handedness": "Right"},
    "08": {"sex": "Male", "Age": 64, "Handedness": "Right"},
    "09": {"sex": "Male", "Age": 57, "Handedness": "Right"},
    "10": {"sex": "Male", "Age": 55, "Handedness": "Right"},
    "11": {"sex": "Male", "Age": 31, "Handedness": "Right"},
    "12": {"sex": "Male", "Age": 58, "Handedness": "Right"},
    "13": {"sex": "Male", "Age": 46, "Handedness": "Right"},
    "14": {"sex": "Female", "Age": 67, "Handedness": "Right"},
    "15": {"sex": "Male", "Age": 63, "Handedness": "left"},
    "16": {"sex": "Male", "Age": 57, "Handedness": "Right"},
    "17": {"sex": "Male", "Age": 60, "Handedness": "Right"},
    "18": {"sex": "Female", "Age": 60, "Handedness": "Right"},
    "19": {"sex": "Female", "Age": 62, "Handedness": "Right"},
    "20": {"sex": "Male", "Age": 34, "Handedness": "Right"},
    "21": {"sex": "Male", "Age": 41, "Handedness": "Right"},
    "22": {"sex": "Male", "Age": 52, "Handedness": "Right"},
    "23": {"sex": "Male", "Age": 57, "Handedness": "Right"},
    "24": {"sex": "Female", "Age": 55, "Handedness": "Right"},
    "25": {"sex": "Male", "Age": 47, "Handedness": "Right"},
    "26": {"sex": "Male", "Age": 61, "Handedness": "Right"},
    "27": {"sex": "Female", "Age": 52, "Handedness": "Right"},
    "28": {"sex": "Female", "Age": 42, "Handedness": "Right"},
    "29": {"sex": "Male", "Age": 53, "Handedness": "Right"},
    "30": {"sex": "Male", "Age": 68, "Handedness": "Right"},
    "31": {"sex": "Female", "Age": 59, "Handedness": "Right"},
    "32": {"sex": "Male", "Age": 74, "Handedness": "Right"},
    "33": {"sex": "Male", "Age": 63, "Handedness": "Right"},
    "34": {"sex": "Female", "Age": 69, "Handedness": "Right"},
    "35": {"sex": "Male", "Age": 69, "Handedness": "Right"},
    "36": {"sex": "Male", "Age": 69, "Handedness": "Right"},
    "37": {"sex": "Male", "Age": 49, "Handedness": "Right"},
    "38": {"sex": "Male", "Age": 53, "Handedness": "Right"},
    "39": {"sex": "Male", "Age": 56, "Handedness": "Right"},
    "40": {"sex": "Female", "Age": 56, "Handedness": "Right"},
    "41": {"sex": "Male", "Age": 77, "Handedness": "Right"},
    "42": {"sex": "Male", "Age": 54, "Handedness": "Right"},
    "43": {"sex": "Male", "Age": 32, "Handedness": "Right"},
    "44": {"sex": "Male", "Age": 59, "Handedness": "Right"},
    "45": {"sex": "Male", "Age": 64, "Handedness": "Right"},
    "46": {"sex": "Male", "Age": 66, "Handedness": "Right"},
    "47": {"sex": "Male", "Age": 40, "Handedness": "Right"},
    "48": {"sex": "Male", "Age": 75, "Handedness": "Right"},
    "49": {"sex": "Male", "Age": 52, "Handedness": "Right"},
    "50": {"sex": "Female", "Age": 64, "Handedness": "Right"},
}


def _resolve_shared_events_file(file_path: str, dataset_path: str | None) -> Path:
    """Locate the shared task-level events TSV shipped with the dataset."""
    candidate_roots = []
    if dataset_path:
        candidate_roots.append(Path(dataset_path))

    file_path = Path(file_path)
    candidate_roots.extend(file_path.parents)

    seen = set()
    for root in candidate_roots:
        if root in seen:
            continue
        seen.add(root)
        candidate = Path(root) / "task-motor-imagery_events.tsv"
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(f"Could not locate task-motor-imagery_events.tsv for {file_path}")


@dataclass
class FigshareStrokeKernel:
    """Stage1 kernel for the figshare-stroke dataset."""

    KERNEL_ID: str = "figshare-stroke-stage1"

    def apply(self, udataset, raw: mne.io.BaseRaw, row) -> mne.io.BaseRaw:
        file_path = row["File Path"]
        shared = udataset.get_shared_attr()
        dataset_path = shared.get("dataset_path") if shared else None

        event_path = _resolve_shared_events_file(file_path, dataset_path)
        event_dataframe = pd.read_csv(event_path, sep="\t")

        subject_id = os.path.basename(file_path).split("-")[1][:2]

        # Build annotations from the events file.
        onsets = []
        durations = []
        descriptions = []
        for _, event in event_dataframe.iterrows():
            start_time = (float(event["onset"]) - 1.0) / 1000.0
            duration = event["duration"] / 1000.0

            if str(event["value"]) == "1":
                label = "Instruction beginning"
            elif str(event["value"]) == "2":
                label = "Motor imagery beginning"
            elif str(event["value"]) == "3":
                label = "Break beginning"
            else:
                label = "Unknown"

            if str(event["trial_type"]) == "1":
                label += " - Left hand"
            elif str(event["trial_type"]) == "2":
                label += " - Right hand"
            else:
                label += " - Unknown"

            # Some shared event files can overshoot the final EDF sample by a
            # couple of milliseconds; clip them instead of dropping the record.
            max_duration = max(raw.times[-1] - start_time, 0.0)
            duration = min(duration, max_duration)
            if duration <= 0.0:
                continue

            onsets.append(start_time)
            durations.append(duration)
            descriptions.append(label)

        raw.set_annotations(
            mne.Annotations(
                onset=np.array(onsets),
                duration=np.array(durations),
                description=np.array(descriptions),
            )
        )

        # Set description with subject info.
        description_dict = {
            "original_description": raw.info.get("description", ""),
            "eegunity_description": {
                "amplifier": "ZhenTec NT1",
                "cap": "ZhenTec NT1'cap",
                "age": str(_SUBJECT_DICT[subject_id]["Age"]),
                "sex": _SUBJECT_DICT[subject_id]["sex"],
                "handedness": _SUBJECT_DICT[subject_id]["Handedness"],
            },
        }
        raw.info["description"] = json.dumps(description_dict)

        # Drop empty channel present in original data.
        if "" in raw.ch_names:
            raw.drop_channels([""])
        type_map = {}
        for ch_name in raw.ch_names:
            if ch_name.upper() in {"HEOL", "HEOR"}:
                type_map[ch_name] = "eog"
        if type_map:
            raw.set_channel_types(type_map, on_unit_change="ignore")

        return raw


KERNEL = FigshareStrokeKernel()
