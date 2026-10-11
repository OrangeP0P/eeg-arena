"""Other MigraineDB stage1 kernel for EEGUnity."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict

import mne
import numpy as np


_SUBJECT_DICT: Dict[str, Dict[str, Any]] = {
    "M1": {"Gender": "Male", "Age": 19},
    "M2": {"Gender": "Female", "Age": 23},
    "M3": {"Gender": "Female", "Age": 19},
    "M4": {"Gender": "Male", "Age": 20},
    "M5": {"Gender": "Female", "Age": 19},
    "M6": {"Gender": "Female", "Age": 54},
    "M7": {"Gender": "Male", "Age": 31},
    "M8": {"Gender": "Male", "Age": 43},
    "M9": {"Gender": "Male", "Age": 31},
    "M10": {"Gender": "Female", "Age": 19},
    "M11": {"Gender": "Female", "Age": 19},
    "M12": {"Gender": "Female", "Age": 22},
    "M13": {"Gender": "Female", "Age": 22},
    "M14": {"Gender": "Female", "Age": 47},
    "M15": {"Gender": "Female", "Age": 20},
    "M16": {"Gender": "Female", "Age": 24},
    "M17": {"Gender": "Female", "Age": 30},
    "M18": {"Gender": "Female", "Age": 30},
    "C1": {"Gender": "Female", "Age": 20},
    "C2": {"Gender": "unknown", "Age": "unknown"},
    "C3": {"Gender": "Female", "Age": 20},
    "C4": {"Gender": "Female", "Age": 19},
    "C5": {"Gender": "Male", "Age": 30},
    "C6": {"Gender": "unknown", "Age": "unknown"},
    "C7": {"Gender": "Male", "Age": 30},
    "C8": {"Gender": "Female", "Age": 30},
    "C9": {"Gender": "Female", "Age": 54},
    "C10": {"Gender": "Male", "Age": 43},
    "C11": {"Gender": "Male", "Age": 24},
    "C12": {"Gender": "unknown", "Age": "unknown"},
    "C13": {"Gender": "Male", "Age": 20},
    "C14": {"Gender": "Female", "Age": 20},
    "C15": {"Gender": "Female", "Age": 20},
    "C16": {"Gender": "Female", "Age": 25},
    "C17": {"Gender": "Female", "Age": 29},
    "C18": {"Gender": "Female", "Age": 26},
    "C19": {"Gender": "Female", "Age": 21},
    "C20": {"Gender": "Female", "Age": 23},
    "C21": {"Gender": "Female", "Age": 49},
}

_MIGRAINE_HEALTH = {"M": "migraine", "C": "Health"}


def _normalize_file_name(file_name: str) -> str:
    """Normalize known filename variants in the released MigraineDB dump."""
    file_name = file_name.lstrip("._")
    file_name = file_name.replace("P15", "M15")
    file_name = re.sub(r"(?i)(M\d+)\s*resting\.bdf$", r"\1_Resting.bdf", file_name)
    file_name = re.sub(r"(?i)(M\d+)\s*ssaep\.bdf$", r"\1_SSAEP.bdf", file_name)
    file_name = re.sub(r"(?i)(M\d+)\s*ssvep\.bdf$", r"\1_SSVEP.bdf", file_name)
    return file_name


def _parse_subject_and_task(file_name: str) -> tuple[str, str]:
    """Return normalized subject ID and task family from a raw BDF basename."""
    normalized = _normalize_file_name(file_name)
    match = re.match(r"^(?P<subject>[MC]\d+)(?:_\d+)?_(?P<task>Resting|SSAEP|SSVEP)(?:_\d+)?\.bdf$", normalized, re.IGNORECASE)
    if not match:
        raise ValueError(f"Unsupported MigraineDB file name: {file_name}")
    subject_id = match.group("subject").upper()
    task_type = match.group("task").upper()
    return subject_id, task_type


def _find_stim_events(raw: mne.io.BaseRaw) -> np.ndarray:
    """Find events from the BioSemi status channel with 1-sample pulses kept."""
    stim_channels = [
        raw.ch_names[i]
        for i in range(len(raw.ch_names))
        if mne.channel_type(raw.info, i) == "stim"
    ]
    if not stim_channels:
        return np.empty((0, 3), dtype=int)
    kwargs = {
        "stim_channel": stim_channels,
        "shortest_event": 1,
        "initial_event": True,
        "verbose": "ERROR",
    }
    try:
        return mne.find_events(raw, event_repeated="merge", **kwargs)
    except TypeError:
        return mne.find_events(raw, **kwargs)


def _set_aux_channel_types(raw: mne.io.BaseRaw) -> None:
    """Restore obvious non-EEG auxiliary channel types from channel names."""
    type_map = {}
    for ch_name in raw.ch_names:
        upper = ch_name.upper()
        if upper == "STATUS":
            type_map[ch_name] = "stim"
        elif upper == "ECG":
            type_map[ch_name] = "ecg"
        elif upper == "RESP":
            type_map[ch_name] = "resp"
        elif upper == "TEMP":
            type_map[ch_name] = "temperature"
        elif upper.startswith("GSR"):
            type_map[ch_name] = "gsr"
        elif upper in {"LO1", "LO2", "IO1", "IO2", "SO1"}:
            type_map[ch_name] = "eog"
    if type_map:
        raw.set_channel_types(type_map, on_unit_change="ignore")


@dataclass
class OtherMigraineDBKernel:
    """Stage1 kernel for the other-migrainedb dataset."""

    KERNEL_ID: str = "other-migrainedb-stage1"

    def apply(self, udataset, raw: mne.io.BaseRaw, row) -> mne.io.BaseRaw:
        file_path = row["File Path"]
        file_name = os.path.basename(file_path)
        subject_id, task_type = _parse_subject_and_task(file_name)

        health_label = _MIGRAINE_HEALTH[subject_id[0]]

        if task_type == "SSVEP":
            event_descriptions = {
                1: f"4Hz vision stimulation start for {health_label}",
                2: f"6Hz vision stimulation start for {health_label}",
                11: f"4Hz vision feedback end for {health_label}",
                22: f"6Hz vision feedback end for {health_label}",
            }
        elif task_type == "SSAEP":
            event_descriptions = {
                1: f"4Hz auditory stimulation start for {health_label}",
                2: f"6Hz auditory stimulation start for {health_label}",
                11: f"4Hz auditory feedback end for {health_label}",
                22: f"6Hz auditory feedback end for {health_label}",
            }
        elif task_type == "RESTING":
            event_descriptions = {
                1: f"resting state block start for {health_label}",
                2: f"resting state block end for {health_label}",
                5: f"resting calibration marker for {health_label}",
            }
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

        description_dict = {
            "original_description": raw.info.get("description", ""),
            "eegunity_description": {
                "amplifier": "BioSemi ActiveTwo system",
                "cap": "custom-designed high-density EEG nylon cap",
                "age": str(_SUBJECT_DICT[subject_id]["Age"]),
                "sex": str(_SUBJECT_DICT[subject_id]["Gender"]),
                "handedness": "unknown",
            },
        }
        raw.info["description"] = json.dumps(description_dict)
        _set_aux_channel_types(raw)

        if task_type == "RESTING":
            recording_duration = raw.times[-1]
            annotations = mne.Annotations(
                onset=[0.0],
                duration=[recording_duration],
                description=[f"resting {health_label}"],
            )
        else:
            events = _find_stim_events(raw)
            valid_ids = sorted(set(event_descriptions).intersection(set(events[:, 2].tolist())))
            if not valid_ids:
                raise ValueError(f"No known stim events found in {file_name}")

            filtered_events = events[np.isin(events[:, 2], valid_ids)]
            annotations = mne.annotations_from_events(
                events=filtered_events,
                sfreq=raw.info["sfreq"],
                event_desc={event_id: event_descriptions[event_id] for event_id in valid_ids},
            )

        raw.set_annotations(annotations)

        return raw


KERNEL = OtherMigraineDBKernel()
