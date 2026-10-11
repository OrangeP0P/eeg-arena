"""TUH EEG Abnormal stage1 kernel for EEGUnity."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import mne


@dataclass
class TuhEegAbnormalKernel:
    """Stage1 kernel for the tuh-eeg-abnormal dataset."""

    KERNEL_ID: str = "tuh-eeg-abnormal-stage1"

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

        # Determine annotation from parent directory (abnormal/normal).
        parent_dir = os.path.basename(
            os.path.dirname(os.path.dirname(file_path))
        )

        if parent_dir == "abnormal":
            description = "abnormal"
        elif parent_dir == "normal":
            description = "normal"
        else:
            description = "unknown"

        onset = 0
        duration = raw.times[-1] - raw.times[0]
        annotations = mne.Annotations(
            onset=[onset], duration=[duration], description=[description]
        )
        raw.set_annotations(annotations)

        return raw


KERNEL = TuhEegAbnormalKernel()
