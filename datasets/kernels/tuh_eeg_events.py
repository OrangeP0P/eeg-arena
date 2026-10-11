"""TUH EEG Events stage1 kernel for EEGUnity."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import mne


_LABEL_MAP = {
    1: "spike and sharp wave",
    2: "generalized periodic epileptiform discharges",
    3: "periodic lateralized epileptiform discharges",
    4: "eye movement",
    5: "artifact",
    6: "background",
}


@dataclass
class TuhEegEventsKernel:
    """Stage1 kernel for the tuh-eeg-events dataset.

    Note: The original script only processes .edf files. Non-.edf files are
    returned unchanged.
    """

    KERNEL_ID: str = "tuh-eeg-events-stage1"

    def apply(self, udataset, raw: mne.io.BaseRaw, row) -> mne.io.BaseRaw:
        file_path = row["File Path"]

        # Only process .edf files (original script filters on '.edf' in File Path).
        if ".edf" not in file_path:
            return raw

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

        # Parse .rec file for event labels.
        rec_file_path = file_path.replace(".edf", ".rec")
        annotations = []

        try:
            with open(rec_file_path, "r", encoding="ISO-8859-1") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) != 4:
                        continue

                    channel, start, stop, label_code = parts
                    start_time = float(start)
                    duration = float(stop) - start_time
                    description = _LABEL_MAP.get(int(label_code), "unknown")
                    annotations.append([start_time, duration, description])

            if annotations:
                onset, duration, description = zip(*annotations)
                raw.set_annotations(
                    mne.Annotations(onset=onset, duration=duration, description=description)
                )

        except FileNotFoundError:
            pass

        return raw


KERNEL = TuhEegEventsKernel()
