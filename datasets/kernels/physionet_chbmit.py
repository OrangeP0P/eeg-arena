"""Physionet CHB-MIT stage1 kernel for EEGUnity."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict

import mne


_SUBJECT_DICT: Dict[str, Dict[str, Any]] = {
    "chb01": {"Gender": "Female", "Age": 11},
    "chb02": {"Gender": "Male", "Age": 11},
    "chb03": {"Gender": "Female", "Age": 14},
    "chb04": {"Gender": "Male", "Age": 22},
    "chb05": {"Gender": "Female", "Age": 7},
    "chb06": {"Gender": "Female", "Age": 1.5},
    "chb07": {"Gender": "Female", "Age": 14.5},
    "chb08": {"Gender": "Male", "Age": 3.5},
    "chb09": {"Gender": "Female", "Age": 10},
    "chb10": {"Gender": "Male", "Age": 3},
    "chb11": {"Gender": "Female", "Age": 12},
    "chb12": {"Gender": "Female", "Age": 2},
    "chb13": {"Gender": "Female", "Age": 3},
    "chb14": {"Gender": "Female", "Age": 9},
    "chb15": {"Gender": "Male", "Age": 16},
    "chb16": {"Gender": "Female", "Age": 7},
    "chb17a": {"Gender": "Female", "Age": 12},
    "chb17b": {"Gender": "Female", "Age": 12},
    "chb17c": {"Gender": "Female", "Age": 12},
    "chb18": {"Gender": "Female", "Age": 18},
    "chb19": {"Gender": "Female", "Age": 19},
    "chb20": {"Gender": "Female", "Age": 6},
    "chb21": {"Gender": "Female", "Age": 13},
    "chb22": {"Gender": "Female", "Age": 9},
    "chb23": {"Gender": "Female", "Age": 6},
    "chb24": {"Gender": "Unknown", "Age": "Unknown"},
}


@dataclass
class PhysionetChbmitKernel:
    """Stage1 kernel for the physionet-chbmit dataset."""

    KERNEL_ID: str = "physionet-chbmit-stage1"

    def apply(self, udataset, raw: mne.io.BaseRaw, row) -> mne.io.BaseRaw:
        file_path = row["File Path"]
        file_name = os.path.basename(file_path)
        subject_id = file_name.split("_")[0]

        description_dict = {
            "original_description": raw.info.get("description", ""),
            "eegunity_description": {
                "amplifier": "unknown",
                "cap": "unknown",
                "age": str(_SUBJECT_DICT[subject_id]["Age"]),
                "sex": str(_SUBJECT_DICT[subject_id]["Gender"]),
                "handedness": "unknown",
            },
        }
        raw.info["description"] = json.dumps(description_dict)

        # Locate the summary file.
        parent_dir = os.path.dirname(file_path)
        summary_file = os.path.join(
            parent_dir, f"{os.path.basename(parent_dir)}-summary.txt"
        )

        onset_list = []
        duration_list = []
        description_list = []

        if os.path.exists(summary_file):
            with open(summary_file, "r") as f:
                summary_data = f.readlines()

            try:
                file_index = next(
                    i
                    for i, line in enumerate(summary_data)
                    if line.startswith("File Name:") and file_name in line
                )

                num_seizures_line = summary_data[file_index + 3]
                if num_seizures_line.startswith("Number of Seizures in File"):
                    num_seizures = int(num_seizures_line.split(":")[-1].strip())
                else:
                    num_seizures = 0

                if num_seizures > 0:
                    for j in range(num_seizures):
                        start_line = summary_data[file_index + 4 + 2 * j]
                        end_line = summary_data[file_index + 5 + 2 * j]

                        start_time = float(
                            start_line.split(":")[-1].strip().replace(" seconds", "")
                        )
                        end_time = float(
                            end_line.split(":")[-1].strip().replace(" seconds", "")
                        )

                        onset_list.append(start_time)
                        duration_list.append(end_time - start_time)
                        description_list.append("seizure")

            except StopIteration:
                pass

        # Compute interictal intervals: complement of all seizure intervals within recording.
        recording_duration = raw.times[-1]
        seizure_intervals = sorted(
            zip(onset_list, [o + d for o, d in zip(onset_list, duration_list)])
        )
        # Merge overlapping seizure intervals.
        merged_seizures = []
        for start, end in seizure_intervals:
            if merged_seizures and start <= merged_seizures[-1][1]:
                merged_seizures[-1] = (merged_seizures[-1][0], max(merged_seizures[-1][1], end))
            else:
                merged_seizures.append((start, end))
        # Build interictal spans as complement of merged seizure spans.
        prev_end = 0.0
        for s_start, s_end in merged_seizures:
            if s_start - prev_end >= 2.0:
                onset_list.append(prev_end)
                duration_list.append(s_start - prev_end)
                description_list.append("interictal")
            prev_end = s_end
        if recording_duration - prev_end >= 2.0:
            onset_list.append(prev_end)
            duration_list.append(recording_duration - prev_end)
            description_list.append("interictal")
        # If no seizures at all, mark the entire recording as interictal.
        if not merged_seizures and recording_duration >= 2.0:
            onset_list.append(0.0)
            duration_list.append(recording_duration)
            description_list.append("interictal")

        if onset_list:
            annotations = mne.Annotations(
                onset=onset_list, duration=duration_list, description=description_list
            )
            raw.set_annotations(annotations)

        return raw


KERNEL = PhysionetChbmitKernel()
