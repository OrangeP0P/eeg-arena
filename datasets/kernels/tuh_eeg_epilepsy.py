"""TUH EEG Epilepsy stage1 kernel for EEGUnity."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import mne


def _get_n_level_parent_dir(file_path, n):
    """Get the directory name n levels up from file_path."""
    parent_dir = file_path
    for _ in range(n):
        parent_dir = os.path.dirname(parent_dir)
    return os.path.basename(parent_dir)


@dataclass
class TuhEegEpilepsyKernel:
    """Stage1 kernel for the tuh-eeg-epilepsy dataset."""

    KERNEL_ID: str = "tuh-eeg-epilepsy-stage1"

    def apply(self, udataset, raw: mne.io.BaseRaw, row) -> mne.io.BaseRaw:
        file_path = row["File Path"]

        sex_mapping = {0: "unknown", 1: "male", 2: "female"}
        description_dict = {
            "original_description": raw.info.get("description", ""),
            "eegunity_description": {
                "amplifier": "unknown",
                "cap": "unknown",
                "age": "unknown",
                "sex": sex_mapping.get(raw.info.get("sex", 0), "unknown"),
                "handedness": "unknown",
            },
        }
        raw.info["description"] = json.dumps(description_dict)

        # Determine annotation from 4-level parent directory.
        parent_dir = _get_n_level_parent_dir(file_path, 4)

        if parent_dir == "00_epilepsy":
            description = "epilepsy"
        elif parent_dir == "01_no_epilepsy":
            description = "no epilepsy"
        else:
            description = "unknown"

        onset = 0
        duration = raw.times[-1] - raw.times[0]
        annotations = mne.Annotations(
            onset=[onset], duration=[duration], description=[description]
        )
        raw.set_annotations(annotations)

        return raw


KERNEL = TuhEegEpilepsyKernel()
