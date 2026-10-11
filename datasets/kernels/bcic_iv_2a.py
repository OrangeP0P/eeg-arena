"""BCIC IV 2a stage1 kernel for EEGUnity.

This kernel updates metadata and events in-memory without saving new files.
It is intended to be loaded externally by EEGUnity.

Spec example
------------
"/path/to/bcic_iv_2a_kernel.py"
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict

import mne
import numpy as np
import scipy.io as scio


# Annotator: Human, 2026-04-08
_SUBJECT_DICT: Dict[str, Dict[str, Any]] = {
    "A01": {"sex": "female", "age": "22"},
    "A02": {"sex": "female", "age": "24"},
    "A03": {"sex": "male",   "age": "26"},
    "A04": {"sex": "female", "age": "24"},
    "A05": {"sex": "male",   "age": "24"},
    "A06": {"sex": "female", "age": "23"},
    "A07": {"sex": "male",   "age": "25"},
    "A08": {"sex": "male",   "age": "23"},
    "A09": {"sex": "male",   "age": "17"},
}
# Source: BCI Competition IV 2a electrode layout, desc_2a.pdf Table 1
# EEG-Fz/C3/Cz/C4/Pz retain their names in GDF; numbered channels EEG-0..EEG-16
# correspond to FC3..P2 in order. EEG-Pz maps to POz (position 22 in the cap).
# Annotator: Human, 2026-04-08
_EEG_CHANNEL_NAMES = {
    "Fz",
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
    "P1",
    "POz",
    "Pz",
    "P2",
}


def _build_channel_rename_map(raw: mne.io.BaseRaw) -> Dict[str, str]:
    """Build a rename map compatible with both raw GDF and locator-normalized names."""
    alias_groups = [
        ("Fz", ["EEG-Fz", "Fz"]),
        ("FC3", ["EEG-0", "0"]),
        ("FC1", ["EEG-1", "1"]),
        ("FCz", ["EEG-2", "2"]),
        ("FC2", ["EEG-3", "3"]),
        ("FC4", ["EEG-4", "4"]),
        ("C5", ["EEG-5", "5"]),
        ("C3", ["EEG-C3", "C3"]),
        ("C1", ["EEG-6", "6"]),
        ("Cz", ["EEG-Cz", "Cz"]),
        ("C2", ["EEG-7", "7"]),
        ("C4", ["EEG-C4", "C4"]),
        ("C6", ["EEG-8", "8"]),
        ("CP3", ["EEG-9", "9"]),
        ("CP1", ["EEG-10", "10"]),
        ("CPz", ["EEG-11", "11"]),
        ("CP2", ["EEG-12", "12"]),
        ("CP4", ["EEG-13", "13"]),
        ("P1", ["EEG-14", "14"]),
        ("POz", ["EEG-Pz", "Pz"]),
        ("Pz", ["EEG-15", "15"]),
        ("P2", ["EEG-16", "16"]),
        ("EOG-left", ["EOG-left", "-left"]),
        ("EOG-central", ["EOG-central", "-central"]),
        ("EOG-right", ["EOG-right", "-right"]),
    ]

    rename_map: Dict[str, str] = {}
    existing_names = set(raw.ch_names)
    for target_name, aliases in alias_groups:
        for source_name in aliases:
            if source_name in existing_names and source_name != target_name:
                rename_map[source_name] = target_name
                break

    return rename_map


def _remap_events(
    original_events: np.ndarray,
    original_event_id: Dict[str, int],
    new_event_id: Dict[str, int],
    numeric_event_map: Dict[str, str],
) -> np.ndarray:
    """Map raw BCIC event annotations to the kernel's canonical event IDs.

    BCIC IV 2a GDF files often expose annotation descriptions as numeric event
    codes (for example ``"769"``) instead of human-readable strings. This
    helper accepts either representation and rewrites matching events to the
    canonical IDs expected downstream.
    """
    remapped_events = original_events.copy()
    old_to_new: Dict[int, int] = {}

    for original_desc, original_id in original_event_id.items():
        canonical_desc = numeric_event_map.get(original_desc, original_desc)
        new_id = new_event_id.get(canonical_desc)
        if new_id is not None:
            old_to_new[original_id] = new_id

    for old_id, new_id in old_to_new.items():
        remapped_events[remapped_events[:, 2] == old_id, 2] = new_id

    return remapped_events


@dataclass
class BCICIV2aKernel:
    """Stage1 kernel for the BCIC IV 2a dataset."""

    KERNEL_ID: str = "bcic-iv-2a-stage1"

    def apply(self, udataset, raw: mne.io.BaseRaw, row) -> mne.io.BaseRaw:
        """Apply stage1 preprocessing to a single record.

        Parameters
        ----------
        udataset
            EEGUnity dataset context (shared-attr holder).
        raw
            In-memory MNE Raw object.
        row
            Locator row (pandas.Series-like) with at least ``File Path``.

        Returns
        -------
        mne.io.BaseRaw
            The modified raw object.
        """
        file_path = row["File Path"]
        subject_id = os.path.basename(file_path)[:3]
        if subject_id not in _SUBJECT_DICT:
            raise ValueError(f"Unrecognized subject id: {subject_id}")

        # 1) Write subject + device meta into raw.info['description'].
        description_dict = {
            "original_description": raw.info.get("description", ""),
            "eegunity_description": {
                "amplifier": "unknown",
                "cap": "Ag/AgCl",
                "age": _SUBJECT_DICT[subject_id]["age"],
                "sex": _SUBJECT_DICT[subject_id]["sex"],
                "handedness": "unknown",
            },
        }
        raw.info["description"] = json.dumps(description_dict)

        # 2) Rename channels + montage + eog types.
        channel_rename_map = _build_channel_rename_map(raw)
        if channel_rename_map:
            raw.rename_channels(channel_rename_map)

        channel_type_map = {
            ch_name: "eeg"
            for ch_name in _EEG_CHANNEL_NAMES
            if ch_name in raw.ch_names
        }
        channel_type_map.update(
            {
                ch_name: "eog"
                for ch_name in ("EOG-left", "EOG-central", "EOG-right")
                if ch_name in raw.ch_names
            }
        )
        if channel_type_map:
            raw.set_channel_types(channel_type_map)

        montage = mne.channels.make_standard_montage("standard_1020")
        raw.info.set_montage(montage, on_missing="ignore")

        # 3) Re-map events.
        event_id = {
            "Rejected trial": 1,
            "Eye movements": 2,
            "Idling EEG (eyes open)": 3,
            "Idling EEG (eyes closed)": 4,
            "Start of a new run": 5,
            "Start of a trial": 6,
            "Cue onset left (class 1)": 7,
            "Cue onset right (class 2)": 8,
            "Cue onset foot (class 3)": 9,
            "Cue onset tongue (class 4)": 10,
        }
        numeric_event_map = {
            "1023": "Rejected trial",
            "1072": "Eye movements",
            "276": "Idling EEG (eyes open)",
            "277": "Idling EEG (eyes closed)",
            "32766": "Start of a new run",
            "768": "Start of a trial",
            "769": "Cue onset left (class 1)",
            "770": "Cue onset right (class 2)",
            "771": "Cue onset foot (class 3)",
            "772": "Cue onset tongue (class 4)",
            "783": "Cue onset left (class 1)",
        }

        events, original_event_id = mne.events_from_annotations(raw)
        events = _remap_events(
            events,
            original_event_id,
            event_id,
            numeric_event_map,
        )

        # 4) If file ends with 'E.gdf', use adjacent .mat classlabel to refine events.
        base, ext = os.path.splitext(file_path)
        if base.endswith("E") and ext.lower() == ".gdf":
            mat_filepath = f"{base}.mat"
            if os.path.exists(mat_filepath):
                mat_data = scio.loadmat(mat_filepath)
                values_from_mat = mat_data["classlabel"].flatten() + 6
                replacement_indices = np.where(events[:, 2] == 7)[0]
                if len(replacement_indices) >= len(values_from_mat):
                    events[replacement_indices[: len(values_from_mat)], 2] = values_from_mat
                else:
                    raise ValueError(
                        f"Not enough cue events to replace using {mat_filepath}."
                    )

        # 5) Convert back to annotations (descriptions are from the mapping).
        id_to_desc = {v: k for k, v in event_id.items()}
        annotations = mne.annotations_from_events(
            events=events,
            sfreq=raw.info["sfreq"],
            event_desc=id_to_desc,
        )
        raw.set_annotations(annotations)
        return raw


KERNEL = BCICIV2aKernel()
