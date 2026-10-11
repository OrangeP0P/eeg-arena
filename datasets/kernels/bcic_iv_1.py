"""BCIC IV 1 stage1 kernel for EEGUnity."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import mne
import scipy.io as scio


@dataclass
class BCICIV1Kernel:
    """Stage1 kernel for the BCIC IV 1 dataset."""

    KERNEL_ID: str = "bcic-iv-1-stage1"

    def apply(self, udataset, raw: mne.io.BaseRaw, row) -> mne.io.BaseRaw:
        file_path = row["File Path"]

        # Update description.
        description_dict = {
            "original_description": raw.info.get("description", ""),
            "eegunity_description": {
                "amplifier": "BrainAmp MR plus",
                "cap": "Ag/AgCl",
                "age": "unknown",
                "sex": "unknown",
                "handedness": "unknown",
            },
        }
        raw.info["description"] = json.dumps(description_dict)

        # Load .mat to get pos/label and create annotations.
        other_data = scio.loadmat(file_path)
        pos = other_data["mrk"][0][0][0].flatten()
        label = other_data["mrk"][0][0][1].flatten()

        class_1_name = other_data["nfo"][0][0][1][0][0][0]
        class_2_name = other_data["nfo"][0][0][1][0][1][0]

        event_map = {-1: class_1_name, 1: class_2_name}
        annotations = mne.Annotations(
            onset=pos / raw.info["sfreq"],
            duration=[0] * len(pos),
            description=[event_map[l] for l in label],
        )
        raw.set_annotations(annotations)

        return raw


KERNEL = BCICIV1Kernel()
