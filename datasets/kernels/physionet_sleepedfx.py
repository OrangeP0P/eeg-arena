"""Physionet Sleep-EDF Expanded stage1 kernel for EEGUnity."""

from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict

import mne


_SC_SUBJECT_DICT: Dict[int, Dict[str, Any]] = {
    0: {"Gender": "Female", "Age": 33},
    1: {"Gender": "Female", "Age": 33},
    2: {"Gender": "Female", "Age": 26},
    3: {"Gender": "Female", "Age": 26},
    4: {"Gender": "Female", "Age": 34},
    5: {"Gender": "Female", "Age": 28},
    6: {"Gender": "Female", "Age": 31},
    7: {"Gender": "Female", "Age": 30},
    8: {"Gender": "Female", "Age": 25},
    9: {"Gender": "Female", "Age": 25},
    10: {"Gender": "Male", "Age": 26},
    11: {"Gender": "Male", "Age": 26},
    12: {"Gender": "Male", "Age": 26},
    13: {"Gender": "Male", "Age": 27},
    14: {"Gender": "Male", "Age": 27},
    15: {"Gender": "Male", "Age": 31},
    16: {"Gender": "Male", "Age": 32},
    17: {"Gender": "Male", "Age": 31},
    18: {"Gender": "Male", "Age": 28},
    19: {"Gender": "Male", "Age": 28},
    20: {"Gender": "Female", "Age": 51},
    21: {"Gender": "Female", "Age": 51},
    22: {"Gender": "Female", "Age": 56},
    23: {"Gender": "Female", "Age": 50},
    24: {"Gender": "Female", "Age": 54},
    25: {"Gender": "Female", "Age": 56},
    26: {"Gender": "Female", "Age": 51},
    27: {"Gender": "Female", "Age": 54},
    28: {"Gender": "Female", "Age": 56},
    29: {"Gender": "Female", "Age": 51},
    30: {"Gender": "Male", "Age": 50},
    31: {"Gender": "Male", "Age": 54},
    32: {"Gender": "Male", "Age": 57},
    33: {"Gender": "Male", "Age": 60},
    34: {"Gender": "Male", "Age": 54},
    35: {"Gender": "Male", "Age": 57},
    36: {"Gender": "Male", "Age": 51},
    37: {"Gender": "Male", "Age": 52},
    38: {"Gender": "Male", "Age": 51},
    40: {"Gender": "Female", "Age": 67},
    41: {"Gender": "Female", "Age": 66},
    42: {"Gender": "Female", "Age": 69},
    43: {"Gender": "Female", "Age": 73},
    44: {"Gender": "Female", "Age": 74},
    45: {"Gender": "Female", "Age": 66},
    46: {"Gender": "Female", "Age": 66},
    47: {"Gender": "Female", "Age": 73},
    48: {"Gender": "Female", "Age": 67},
    49: {"Gender": "Female", "Age": 67},
    50: {"Gender": "Male", "Age": 71},
    51: {"Gender": "Male", "Age": 70},
    52: {"Gender": "Male", "Age": 69},
    53: {"Gender": "Male", "Age": 67},
    54: {"Gender": "Male", "Age": 73},
    55: {"Gender": "Male", "Age": 71},
    56: {"Gender": "Male", "Age": 72},
    57: {"Gender": "Male", "Age": 66},
    58: {"Gender": "Male", "Age": 67},
    59: {"Gender": "Male", "Age": 67},
    60: {"Gender": "Female", "Age": 89},
    61: {"Gender": "Female", "Age": 101},
    62: {"Gender": "Female", "Age": 95},
    63: {"Gender": "Female", "Age": 91},
    64: {"Gender": "Female", "Age": 85},
    65: {"Gender": "Female", "Age": 88},
    66: {"Gender": "Female", "Age": 88},
    67: {"Gender": "Female", "Age": 87},
    70: {"Gender": "Male", "Age": 89},
    71: {"Gender": "Male", "Age": 88},
    72: {"Gender": "Male", "Age": 88},
    73: {"Gender": "Male", "Age": 97},
    74: {"Gender": "Male", "Age": 92},
    75: {"Gender": "Male", "Age": 96},
    76: {"Gender": "Male", "Age": 90},
    77: {"Gender": "Male", "Age": 85},
    80: {"Gender": "Female", "Age": 54},
    81: {"Gender": "Female", "Age": 57},
    82: {"Gender": "Female", "Age": 56},
}

_ST_SUBJECT_DICT: Dict[int, Dict[str, Any]] = {
    1: {"Gender": "Male", "Age": 60},
    2: {"Gender": "Female", "Age": 35},
    4: {"Gender": "Female", "Age": 18},
    5: {"Gender": "Female", "Age": 32},
    6: {"Gender": "Female", "Age": 35},
    7: {"Gender": "Female", "Age": 51},
    8: {"Gender": "Female", "Age": 66},
    9: {"Gender": "Male", "Age": 47},
    10: {"Gender": "Female", "Age": 20},
    11: {"Gender": "Female", "Age": 21},
    12: {"Gender": "Male", "Age": 21},
    13: {"Gender": "Male", "Age": 22},
    14: {"Gender": "Male", "Age": 20},
    15: {"Gender": "Female", "Age": 66},
    16: {"Gender": "Female", "Age": 79},
    17: {"Gender": "Female", "Age": 48},
    18: {"Gender": "Female", "Age": 53},
    19: {"Gender": "Female", "Age": 28},
    20: {"Gender": "Male", "Age": 24},
    21: {"Gender": "Female", "Age": 34},
    22: {"Gender": "Male", "Age": 56},
    24: {"Gender": "Female", "Age": 48},
}


@dataclass
class PhysionetSleepedfxKernel:
    """Stage1 kernel for the physionet-sleepedfx dataset.

    Note: The original script only processes PSG files. Non-PSG files are
    returned unchanged (no description set) so the caller can filter them.
    """

    KERNEL_ID: str = "physionet-sleepedfx-stage1"

    def apply(self, udataset, raw: mne.io.BaseRaw, row) -> mne.io.BaseRaw:
        file_path = row["File Path"]
        basename = os.path.basename(file_path)

        # Only process PSG files (original script filters on 'PSG' in File Path).
        if "PSG" not in file_path:
            return raw

        # Determine SC/ST and get subject info.
        if basename.startswith("SC"):
            subject_id = int(basename[3:5])
            subject_dict = _SC_SUBJECT_DICT
            pattern = r"(SC\d+)(\d{2})(\d)"
        elif basename.startswith("ST"):
            subject_id = int(basename[3:5])
            subject_dict = _ST_SUBJECT_DICT
            pattern = r"(ST\d+)(\d{2})(\d)"
        else:
            return raw

        description_dict = {
            "original_description": raw.info.get("description", ""),
            "eegunity_description": {
                "amplifier": "unknown",
                "cap": "unknown",
                "age": str(subject_dict[subject_id]["Age"]),
                "sex": str(subject_dict[subject_id]["Gender"]),
                "handedness": "unknown",
            },
        }
        raw.info["description"] = json.dumps(description_dict)

        # Find corresponding Hypnogram.edf file.
        match = re.search(pattern, file_path)
        if not match:
            return raw

        dataset_id = match.group(1)
        ss_value = match.group(2)
        n_value = match.group(3)

        event_file_pattern = f"{dataset_id}{ss_value}{n_value}*-Hypnogram.edf"
        event_file_candidates = glob.glob(
            os.path.join(os.path.dirname(file_path), event_file_pattern)
        )

        try:
            event_file_path = event_file_candidates[0]
            annotations = mne.read_annotations(event_file_path)
            raw.set_annotations(annotations)
        except (IndexError, Exception):
            pass

        return raw


KERNEL = PhysionetSleepedfxKernel()
