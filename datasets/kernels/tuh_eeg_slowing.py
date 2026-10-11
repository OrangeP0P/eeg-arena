"""TUH EEG Slowing stage1 kernel for EEGUnity."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mne
import pandas as pd


_REQUIRED_COLUMNS = ["channel", "start_time", "stop_time", "label"]


def _build_description_dict(raw: mne.io.BaseRaw) -> dict:
    sex_mapping = {0: "unknown", 1: "male", 2: "female"}
    return {
        "original_description": raw.info.get("description", ""),
        "eegunity_description": {
            "amplifier": "unknown",
            "cap": "unknown",
            "age": "unknown",
            "sex": sex_mapping.get(raw.info.get("sex", 0), "unknown"),
            "handedness": "unknown",
        },
    }


def _get_sidecar_csv_path(row) -> Path:
    return Path(str(row["File Path"])).with_suffix(".csv")


def _load_annotation_rows(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Annotation CSV not found: {csv_path}")

    annotations_df = pd.read_csv(csv_path, comment="#")
    missing_columns = [col for col in _REQUIRED_COLUMNS if col not in annotations_df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in {csv_path.name}: {missing_columns}")

    annotations_df = annotations_df.dropna(subset=["start_time", "stop_time", "label"]).copy()
    return annotations_df


@dataclass
class TuhEegSlowingKernel:
    """Stage1 kernel for the tuh-eeg-slowing dataset."""

    KERNEL_ID: str = "tuh-eeg-slowing-stage1"

    def apply(self, udataset, raw: mne.io.BaseRaw, row) -> mne.io.BaseRaw:
        file_path = str(row["File Path"])
        if not file_path.lower().endswith(".edf"):
            return raw

        description_dict = _build_description_dict(raw)
        raw.info["description"] = json.dumps(description_dict)

        csv_path = _get_sidecar_csv_path(row)
        annotations_df = _load_annotation_rows(csv_path)

        start_time = annotations_df["start_time"].astype(float).values
        stop_time = annotations_df["stop_time"].astype(float).values
        label = annotations_df["label"].astype(str).values
        duration = stop_time - start_time

        annotations = mne.Annotations(onset=start_time, duration=duration, description=label)
        raw.set_annotations(annotations)

        return raw


KERNEL = TuhEegSlowingKernel()
