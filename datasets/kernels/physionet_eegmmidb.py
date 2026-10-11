"""Physionet EEG Motor Movement/Imagery Database stage1 kernel for EEGUnity."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import mne
from mne import Annotations


@dataclass
class PhysionetEegmmidbKernel:
    """Stage1 kernel for the physionet-eegmmidb dataset."""

    KERNEL_ID: str = "physionet-eegmmidb-stage1"

    def apply(self, udataset, raw: mne.io.BaseRaw, row) -> mne.io.BaseRaw:
        file_path = row["File Path"]
        filename = os.path.basename(file_path)

        description_dict = {
            "original_description": raw.info.get("description", ""),
            "eegunity_description": {
                "amplifier": "unknown",
                "cap": "unknown",
                "age": "unknown",
                "sex": "unknown",
                "handedness": "unknown",
            },
        }
        raw.info["description"] = json.dumps(description_dict)

        # Extract run number from filename (SAAABB.edf -> BB is run number).
        run_number = int(filename[5:7])

        # Define run-specific event descriptions.
        event_descriptions = {"T0": "Rest"}

        if run_number in [3, 7, 11]:
            event_descriptions["T1"] = "Open Left Fist Movement"
            event_descriptions["T2"] = "Open Right Fist Movement"
        elif run_number in [5, 9, 13]:
            event_descriptions["T1"] = "Open Both Fists Movement"
            event_descriptions["T2"] = "Open Both Feet Movement"
        elif run_number in [4, 8, 12]:
            event_descriptions["T1"] = "Imagine Opening Left Fist Movement"
            event_descriptions["T2"] = "Imagine Opening Right Fist Movement"
        elif run_number in [6, 10, 14]:
            event_descriptions["T1"] = "Imagine Opening Both Fists Movement"
            event_descriptions["T2"] = "Imagine Opening Feet Movement"
        elif run_number in [1]:
            event_descriptions["T0"] = "Baseline, eyes open"
        elif run_number in [2]:
            event_descriptions["T0"] = "Baseline, eyes closed"

        # Remap annotation descriptions.
        new_descriptions = [
            event_descriptions.get(desc, desc) for desc in raw.annotations.description
        ]

        new_annotations = Annotations(
            onset=raw.annotations.onset,
            duration=raw.annotations.duration,
            description=new_descriptions,
            orig_time=raw.annotations.orig_time,
        )
        raw.set_annotations(new_annotations)

        return raw


KERNEL = PhysionetEegmmidbKernel()
