"""Physionet CAP Sleep Database stage1 kernel for EEGUnity."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import mne
import pandas as pd


_SUBJECT_DICT: Dict[str, Dict[str, Any]] = {
    "BRUX1": {"Gender": "Male", "Age": 34},
    "BRUX2": {"Gender": "Male", "Age": 23},
    "SDB1": {"Gender": "Male", "Age": 65},
    "SDB2": {"Gender": "Male", "Age": 77},
    "SDB3": {"Gender": "Male", "Age": 78},
    "SDB4": {"Gender": "Male", "Age": 65},
    "INS1": {"Gender": "Male", "Age": 54},
    "INS2": {"Gender": "Female", "Age": 58},
    "INS3": {"Gender": "Male", "Age": 82},
    "INS4": {"Gender": "Female", "Age": 58},
    "INS5": {"Gender": "Female", "Age": 59},
    "INS6": {"Gender": "Female", "Age": 54},
    "INS7": {"Gender": "Female", "Age": 47},
    "INS8": {"Gender": "Male", "Age": 64},
    "INS9": {"Gender": "Male", "Age": 72},
    "NARCO1": {"Gender": "Female", "Age": 29},
    "NARCO2": {"Gender": "Female", "Age": 44},
    "NARCO3": {"Gender": "Female", "Age": 18},
    "NARCO4": {"Gender": "Male", "Age": 43},
    "NARCO5": {"Gender": "Male", "Age": 24},
    "NFLE1": {"Gender": "Female", "Age": 16},
    "NFLE2": {"Gender": "Female", "Age": 41},
    "NFLE3": {"Gender": "Male", "Age": 29},
    "NFLE4": {"Gender": "Male", "Age": 18},
    "NFLE5": {"Gender": "Female", "Age": 22},
    "NFLE6": {"Gender": "Female", "Age": 32},
    "NFLE7": {"Gender": "Male", "Age": 26},
    "NFLE8": {"Gender": "Male", "Age": 14},
    "NFLE9": {"Gender": "Female", "Age": 21},
    "NFLE10": {"Gender": "Male", "Age": 18},
    "NFLE11": {"Gender": "Male", "Age": 31},
    "NFLE12": {"Gender": "Female", "Age": 67},
    "NFLE13": {"Gender": "Female", "Age": 36},
    "NFLE14": {"Gender": "Male", "Age": 35},
    "NFLE15": {"Gender": "Female", "Age": 29},
    "NFLE16": {"Gender": "Female", "Age": 30},
    "NFLE17": {"Gender": "Male", "Age": 25},
    "NFLE18": {"Gender": "Male", "Age": 25},
    "NFLE19": {"Gender": "Male", "Age": 25},
    "NFLE20": {"Gender": "Female", "Age": 26},
    "NFLE21": {"Gender": "Male", "Age": 27},
    "NFLE22": {"Gender": "Female", "Age": 42},
    "NFLE23": {"Gender": "Male", "Age": 20},
    "NFLE24": {"Gender": "Male", "Age": 39},
    "NFLE25": {"Gender": "Male", "Age": 40},
    "NFLE26": {"Gender": "Male", "Age": 38},
    "NFLE27": {"Gender": "Female", "Age": 33},
    "NFLE28": {"Gender": "Female", "Age": 28},
    "NFLE29": {"Gender": "Female", "Age": 36},
    "NFLE30": {"Gender": "Female", "Age": 26},
    "NFLE31": {"Gender": "Male", "Age": 33},
    "NFLE32": {"Gender": "Female", "Age": 34},
    "NFLE33": {"Gender": "Female", "Age": 30},
    "NFLE34": {"Gender": "Male", "Age": 26},
    "NFLE35": {"Gender": "Male", "Age": 44},
    "NFLE36": {"Gender": "Female", "Age": 18},
    "NFLE37": {"Gender": "Male", "Age": 16},
    "NFLE38": {"Gender": "Male", "Age": 31},
    "NFLE39": {"Gender": "Male", "Age": 24},
    "NFLE40": {"Gender": "Female", "Age": 60},
    "PLM1": {"Gender": "Male", "Age": 61},
    "PLM2": {"Gender": "Female", "Age": 52},
    "PLM3": {"Gender": "Female", "Age": 52},
    "PLM4": {"Gender": "Male", "Age": 62},
    "PLM5": {"Gender": "Male", "Age": 40},
    "PLM6": {"Gender": "Male", "Age": 58},
    "PLM7": {"Gender": "Male", "Age": 58},
    "PLM8": {"Gender": "Male", "Age": 60},
    "PLM9": {"Gender": "Male", "Age": 58},
    "PLM10": {"Gender": "Female", "Age": 50},
    "RBD1": {"Gender": "Male", "Age": 58},
    "RBD2": {"Gender": "Male", "Age": 77},
    "RBD3": {"Gender": "Male", "Age": 81},
    "RBD4": {"Gender": "Male", "Age": 70},
    "RBD5": {"Gender": "Female", "Age": 75},
    "RBD6": {"Gender": "Male", "Age": 73},
    "RBD7": {"Gender": "Male", "Age": 72},
    "RBD8": {"Gender": "Male", "Age": 82},
    "RBD9": {"Gender": "Male", "Age": 76},
    "RBD10": {"Gender": "Male", "Age": 73},
    "RBD11": {"Gender": "Male", "Age": 73},
    "RBD12": {"Gender": "Female", "Age": 76},
    "RBD13": {"Gender": "Male", "Age": 61},
    "RBD14": {"Gender": "Male", "Age": 66},
    "RBD15": {"Gender": "Male", "Age": 69},
    "RBD16": {"Gender": "Male", "Age": 65},
    "RBD17": {"Gender": "Male", "Age": 70},
    "RBD18": {"Gender": "Male", "Age": 59},
    "RBD19": {"Gender": "Male", "Age": 65},
    "RBD20": {"Gender": "Male", "Age": 72},
    "RBD21": {"Gender": "Female", "Age": 73},
    "RBD22": {"Gender": "Male", "Age": 70},
    "N1": {"Gender": "Female", "Age": 37},
    "N2": {"Gender": "Male", "Age": 34},
    "N3": {"Gender": "Female", "Age": 35},
    "N4": {"Gender": "Female", "Age": 25},
    "N5": {"Gender": "Female", "Age": 35},
    "N6": {"Gender": "Male", "Age": 31},
    "N7": {"Gender": "Male", "Age": 31},
    "N8": {"Gender": "Female", "Age": 42},
    "N9": {"Gender": "Male", "Age": 31},
    "N10": {"Gender": "Male", "Age": 23},
    "N11": {"Gender": "Female", "Age": 28},
    "N12": {"Gender": "Male", "Age": 29},
    "N13": {"Gender": "Female", "Age": 24},
    "N14": {"Gender": "Female", "Age": 35},
    "N15": {"Gender": "Male", "Age": 34},
    "N16": {"Gender": "Female", "Age": 41},
}


def _calculate_onset(meas_date, time_str):
    """Calculate onset in seconds relative to meas_date."""
    if meas_date.tzinfo is None:
        meas_date = meas_date.replace(tzinfo=timezone.utc)
    else:
        meas_date = meas_date.astimezone(timezone.utc)

    if re.match(r"^\d{1,2}:\d{1,2}:\d{1,2}$", time_str):
        time_str_time = datetime.strptime(time_str, "%H:%M:%S").time()
    elif re.match(r"^\d{1,2}\.\d{1,2}\.\d{1,2}$", time_str):
        time_str_time = datetime.strptime(time_str, "%H.%M.%S").time()
    else:
        raise ValueError(f"Unsupported time format: {time_str}")

    event_time = datetime.combine(meas_date.date(), time_str_time)
    event_time = event_time.replace(tzinfo=timezone.utc)

    if event_time < meas_date:
        event_time += timedelta(days=1)

    return (event_time - meas_date).total_seconds()


def _build_background_segments(cap_segments, recording_duration: float):
    """Construct CAP background ('B') segments as complement of A-phases."""
    merged = []
    for onset, duration in sorted(cap_segments):
        if duration <= 0:
            continue
        end = min(onset + duration, recording_duration)
        if end <= onset:
            continue
        if not merged or onset > merged[-1][1]:
            merged.append([onset, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    background = []
    cursor = 0.0
    for onset, end in merged:
        if onset > cursor:
            background.append((cursor, onset - cursor))
        cursor = max(cursor, end)
    if cursor < recording_duration:
        background.append((cursor, recording_duration - cursor))
    return background


def _duration_epsilon(raw: mne.io.BaseRaw) -> float:
    """Small margin so exact-length long events survive strict '>' checks."""
    return max(1.0 / float(raw.info["sfreq"]), 1e-3)


def _normalize_aux_channels(raw: mne.io.BaseRaw) -> None:
    """Give CAP auxiliary channels stable unique names and better channel types."""
    rename_map = {}
    type_map = {}
    reserved_names = set(raw.ch_names)

    def _reserve_unique_name(original_name: str, preferred_name: str) -> str:
        if original_name == preferred_name:
            return preferred_name

        reserved_names.discard(original_name)
        if preferred_name not in reserved_names:
            reserved_names.add(preferred_name)
            return preferred_name

        suffix = 2
        while True:
            candidate = f"{preferred_name}-{suffix}"
            if candidate not in reserved_names:
                reserved_names.add(candidate)
                return candidate
            suffix += 1

    for ch_name in raw.ch_names:
        upper = ch_name.upper()
        if upper in {"ROC-LOC", "LOC-ROC"}:
            type_map[ch_name] = "eog"
        elif upper in {"EMG1-EMG2", "EMG CHIN"}:
            preferred_name = "EMG-bipolar" if upper == "EMG1-EMG2" else "EMG-chin"
            rename_map[ch_name] = _reserve_unique_name(ch_name, preferred_name)
            type_map[ch_name] = "emg"
        elif upper == "ECG1-ECG2":
            rename_map[ch_name] = _reserve_unique_name(ch_name, "ECG-bipolar")
            type_map[ch_name] = "ecg"
        elif upper in {"DX1-DX2", "SX1-SX2", "TIB SX", "TIB DX"}:
            type_map[ch_name] = "emg"
        elif upper in {"TERMISTORE", "FLUSSO", "TORACE", "ADDOME", "ADDDOME"}:
            type_map[ch_name] = "resp"
        elif upper in {"SAO2", "SPO2", "PLETH", "HR", "STAT", "OX STATUS", "POSITION", "POSIZIONE", "MIC"}:
            type_map[ch_name] = "misc"

    if rename_map:
        raw.rename_channels(rename_map)

    adjusted_type_map = {}
    for ch_name, ch_type in type_map.items():
        target_name = rename_map.get(ch_name, ch_name)
        if target_name in raw.ch_names:
            adjusted_type_map[target_name] = ch_type
    if adjusted_type_map:
        raw.set_channel_types(adjusted_type_map, on_unit_change="ignore")


@dataclass
class PhysionetCapslpdbKernel:
    """Stage1 kernel for the physionet-capslpdb dataset."""

    KERNEL_ID: str = "physionet-capslpdb-stage1"

    def apply(self, udataset, raw: mne.io.BaseRaw, row) -> mne.io.BaseRaw:
        file_path = row["File Path"]
        subject_id = os.path.basename(file_path).split(".")[0].upper()
        _normalize_aux_channels(raw)

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

        # Parse annotation .txt file.
        txt_file_path = file_path.replace(".edf", ".txt")

        data = {
            "Sleep Stage": [],
            "Position": [],
            "Time": [],
            "Event": [],
            "Duration": [],
            "Location": [],
        }

        with open(txt_file_path, "r") as file:
            for line in file:
                if not line.strip() or line.startswith("Patient"):
                    continue

                parts = line.strip().split("\t")

                if len(parts) == 5:
                    data["Sleep Stage"].append(parts[0])
                    data["Position"].append(None)
                    data["Time"].append(parts[1])
                    data["Event"].append(parts[2])
                    data["Duration"].append(parts[3])
                    data["Location"].append(parts[4])
                elif len(parts) >= 6:
                    data["Sleep Stage"].append(parts[0])
                    data["Position"].append(parts[1])
                    data["Time"].append(parts[2])
                    data["Event"].append(parts[3])
                    data["Duration"].append(parts[4])
                    data["Location"].append(parts[5] if len(parts) > 5 else None)

        df = pd.DataFrame(data)

        annotations = []
        cap_segments = []
        eps = _duration_epsilon(raw)
        for _, row_events in df.iterrows():
            time_str = row_events["Time"]
            event_type = row_events["Event"]
            try:
                duration = float(row_events["Duration"])
            except ValueError:
                continue

            onset = _calculate_onset(raw.info["meas_date"], time_str)
            if event_type.startswith("MCAP-A"):
                label = event_type.replace("MCAP-", "")
                onset = max(0.0, onset - 0.5)
                duration = duration + 0.5 + eps
                annotations.append(dict(onset=onset, duration=duration, description=label))
                cap_segments.append((onset, duration))
            elif event_type.startswith("SLEEP-"):
                annotations.append(
                    dict(onset=onset, duration=duration + eps, description=event_type)
                )

        recording_duration = raw.times[-1]
        for onset, duration in _build_background_segments(cap_segments, recording_duration):
            annotations.append(dict(onset=onset, duration=duration, description="B"))

        annotations.sort(key=lambda ann: ann["onset"])

        raw.set_annotations(
            mne.Annotations(
                onset=[ann["onset"] for ann in annotations],
                duration=[ann["duration"] for ann in annotations],
                description=[ann["description"] for ann in annotations],
            )
        )

        return raw


KERNEL = PhysionetCapslpdbKernel()
