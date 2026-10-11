"""Physionet HMC Sleep Staging stage1 kernel for EEGUnity."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import mne
import pandas as pd


def _duration_epsilon(raw: mne.io.BaseRaw) -> float:
    """Small margin so 30s stages pass strict long-event duration checks."""
    return max(1.0 / float(raw.info["sfreq"]), 1e-3)


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
    elif re.match(r"^\d{1,2}\.\d{1,2}\.\d{1,2}\.\d{1,3}$", time_str):
        time_str_time = datetime.strptime(time_str, "%H.%M.%S.%f").time()
    else:
        raise ValueError(f"Unsupported time format: {time_str}")

    event_time = datetime.combine(meas_date.date(), time_str_time)
    event_time = event_time.replace(tzinfo=timezone.utc)

    if event_time < meas_date:
        event_time += timedelta(days=1)

    return (event_time - meas_date).total_seconds()


@dataclass
class PhysionetHmcSleepStagingKernel:
    """Stage1 kernel for the physionet-hmcsleepstaging dataset."""

    KERNEL_ID: str = "physionet-hmcsleepstaging-stage1"

    def apply(self, udataset, raw: mne.io.BaseRaw, row) -> mne.io.BaseRaw:
        file_path = row["File Path"]

        description_dict = {
            "original_description": raw.info.get("description", ""),
            "eegunity_description": {
                "amplifier": "EEG 10-20 recorders (SOMNOmedics, Germany)",
                "cap": "Ag/AgCl",
                "age": "unknown",
                "sex": "unknown",
                "handedness": "unknown",
            },
        }
        raw.info["description"] = json.dumps(description_dict)

        # Special case: SN079 has no annotation file.
        if os.path.basename(file_path) == "SN079.edf":
            return raw

        # Parse _sleepscoring.txt file.
        txt_file_path = file_path.replace(".edf", "_sleepscoring.txt")

        data = {
            "Date": [],
            "Time": [],
            "Recording onset": [],
            "Duration": [],
            "Annotation": [],
            "Linked channel": [],
        }

        with open(txt_file_path, "r") as file:
            for line in file:
                if line.startswith("Date"):
                    continue
                parts = line.strip().split(", ")
                if len(parts) >= 5:
                    data["Date"].append(parts[0])
                    data["Time"].append(parts[1])
                    data["Recording onset"].append(parts[2])
                    data["Duration"].append(parts[3])
                    data["Annotation"].append(parts[4])
                    data["Linked channel"].append(parts[5] if len(parts) > 5 else None)

        df = pd.DataFrame(data)

        annotations = []
        eps = _duration_epsilon(raw)
        for _, row_events in df.iterrows():
            event_type = str(row_events["Annotation"]).strip().rstrip(",")
            if not (str(event_type).startswith("Sleep stage ") or
                    event_type in ("Lights off", "Lights on")):
                continue

            try:
                duration = float(row_events["Duration"])
            except ValueError:
                continue

            try:
                onset = float(row_events["Recording onset"])
            except ValueError:
                onset = _calculate_onset(raw.info["meas_date"], row_events["Time"])
            annotations.append(dict(onset=onset, duration=duration + eps, description=event_type))

        raw.set_annotations(
            mne.Annotations(
                onset=[ann["onset"] for ann in annotations],
                duration=[ann["duration"] for ann in annotations],
                description=[ann["description"] for ann in annotations],
            )
        )

        return raw


KERNEL = PhysionetHmcSleepStagingKernel()
