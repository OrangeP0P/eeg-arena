"""OpenNeuro stage1 kernel for EEGUnity.

A unified kernel for all OpenNeuro datasets (ds005505-ds005516). The
amplifier/cap are passed per-dataset via the constructor. If not provided,
defaults to 'unknown'.

The kernel parses participants.tsv from the dataset root and dynamically
extracts all available columns for each subject.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import mne
import pandas as pd


# Per-dataset amplifier/cap configuration.
_DEVICE_MAP = {
    "openneuro-ds005505": ("GSN 200, Magnetism-EGI, Eugene, OR", "GSN 200, Magnetism-EGI, Eugene, OR"),
    "openneuro-ds005506": ("GSN 200, Magnetism-EGI, Eugene, OR", "GSN 200, Magnetism-EGI, Eugene, OR"),
    "openneuro-ds005507": ("GSN 200, Magnetism-EGI, Eugene, OR", "GSN 200, Magnetism-EGI, Eugene, OR"),
    "openneuro-ds005508": ("EGI Net Amps 400", "EGI GSN 200"),
    "openneuro-ds005509": ("EGI Net Amps 400", "EGI GSN 200"),
    "openneuro-ds005510": ("EGI Net Amps 400", "EGI GSN 200"),
    "openneuro-ds005511": ("EGI Net Amps 400", "EGI GSN 200"),
    "openneuro-ds005512": ("EGI Net Amps 400", "EGI GSN 200"),
    "openneuro-ds005513": ("EGI Net Amps 400", "EGI GSN 200"),
    "openneuro-ds005514": ("EGI Net Amps 400", "EGI GSN 200"),
    "openneuro-ds005515": ("EGI Net Amps 400", "EGI GSN 200"),
    "openneuro-ds005516": ("EGI Net Amps 400", "EGI GSN 200"),
}

# Standard columns always present in the description.
_STANDARD_KEYS = {"age", "sex", "handedness"}


@dataclass
class OpenNeuroKernel:
    """Stage1 kernel for OpenNeuro EEG datasets."""

    KERNEL_ID: str = "openneuro-stage1"

    # Cache for participants DataFrames keyed by dataset_path.
    _participants_cache: dict = field(default_factory=dict, repr=False)

    def _load_participants(self, dataset_path: str) -> pd.DataFrame | None:
        """Find and load participants.tsv from the dataset root."""
        if dataset_path in self._participants_cache:
            return self._participants_cache[dataset_path]

        tsv_path = None
        for root, dirs, files in os.walk(dataset_path):
            if "participants.tsv" in files:
                tsv_path = os.path.join(root, "participants.tsv")
                break

        if tsv_path is None:
            self._participants_cache[dataset_path] = None
            return None

        df = pd.read_csv(tsv_path, sep="\t")
        self._participants_cache[dataset_path] = df
        return df

    def apply(self, udataset, raw: mne.io.BaseRaw, row) -> mne.io.BaseRaw:
        file_path = row["File Path"]
        dataset_path = udataset.get_shared_attr()["dataset_path"]
        domain_tag = udataset.get_shared_attr().get("domain_tag", "")

        # Determine amplifier/cap.
        amplifier, cap = _DEVICE_MAP.get(domain_tag, ("unknown", "unknown"))

        # Load participants.
        participants_df = self._load_participants(dataset_path)

        # Extract subject id from directory structure (two levels up).
        subject_id = os.path.basename(
            os.path.dirname(os.path.dirname(file_path))
        )

        eeg_desc = {
            "amplifier": amplifier,
            "cap": cap,
            "age": "unknown",
            "sex": "unknown",
            "handedness": "unknown",
        }

        if participants_df is not None:
            subject_info = participants_df[
                participants_df["participant_id"] == subject_id
            ]
            if not subject_info.empty:
                # Extract standard fields.
                if "age" in subject_info.columns:
                    eeg_desc["age"] = str(subject_info["age"].values[0])
                if "sex" in subject_info.columns:
                    sex_val = subject_info["sex"].values[0]
                    if sex_val == "F":
                        eeg_desc["sex"] = "Female"
                    elif sex_val == "M":
                        eeg_desc["sex"] = "Male"
                    else:
                        eeg_desc["sex"] = str(sex_val)

                # Dynamically extract all other columns.
                for col in subject_info.columns:
                    if col == "participant_id":
                        continue
                    if col in _STANDARD_KEYS:
                        continue
                    eeg_desc[col] = str(subject_info[col].values[0])

        description_dict = {
            "original_description": raw.info.get("description", ""),
            "eegunity_description": eeg_desc,
        }
        raw.info["description"] = json.dumps(description_dict)

        return raw


KERNEL = OpenNeuroKernel()
