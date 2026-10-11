"""TUH EEG Artifact stage1 kernel for EEGUnity."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mne
import pandas as pd

from eegunity.utils.annotation_metadata import encode_annotation_description


_REQUIRED_COLUMNS = ["channel", "start_time", "stop_time", "label"]
_MIN_DURATION = 1e-6


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
    annotations_df["start_time"] = pd.to_numeric(annotations_df["start_time"], errors="coerce")
    annotations_df["stop_time"] = pd.to_numeric(annotations_df["stop_time"], errors="coerce")
    annotations_df["label"] = annotations_df["label"].astype(str).str.strip()
    annotations_df = annotations_df.dropna(subset=["start_time", "stop_time", "label"])
    annotations_df = annotations_df[annotations_df["stop_time"] > annotations_df["start_time"]]

    # TUAR annotations are repeated per channel; tasks here are interval-level.
    return (
        annotations_df
        .groupby(["start_time", "stop_time", "label"], as_index=False)
        .agg(channel_count=("channel", "nunique"))
        .sort_values(["start_time", "stop_time", "label"])
        .reset_index(drop=True)
    )


def _recording_duration(raw: mne.io.BaseRaw) -> float:
    sfreq = float(raw.info.get("sfreq") or 0.0)
    if sfreq > 0 and raw.n_times:
        return float(raw.n_times / sfreq)
    if len(raw.times):
        return float(raw.times[-1])
    return 0.0


def _build_background_rows(annotations_df: pd.DataFrame, recording_duration: float) -> list[tuple[float, float]]:
    if recording_duration <= 0:
        return []

    intervals = []
    for row in annotations_df.itertuples(index=False):
        start = max(0.0, float(row.start_time))
        stop = min(recording_duration, float(row.stop_time))
        if stop - start > _MIN_DURATION:
            intervals.append((start, stop))
    if not intervals:
        return [(0.0, recording_duration)]

    intervals.sort()
    merged = []
    current_start, current_stop = intervals[0]
    for start, stop in intervals[1:]:
        if start <= current_stop:
            current_stop = max(current_stop, stop)
            continue
        merged.append((current_start, current_stop))
        current_start, current_stop = start, stop
    merged.append((current_start, current_stop))

    background = []
    cursor = 0.0
    for start, stop in merged:
        if start - cursor > _MIN_DURATION:
            background.append((cursor, start))
        cursor = max(cursor, stop)
    if recording_duration - cursor > _MIN_DURATION:
        background.append((cursor, recording_duration))
    return background


@dataclass
class TuhEegArtifactKernel:
    """Stage1 kernel for the tuh-eeg-artifact dataset."""

    KERNEL_ID: str = "tuh-eeg-artifact-stage1"

    def apply(self, udataset, raw: mne.io.BaseRaw, row) -> mne.io.BaseRaw:
        file_path = str(row["File Path"])
        if not file_path.lower().endswith(".edf"):
            return raw

        raw.info["description"] = json.dumps(_build_description_dict(raw))

        csv_path = _get_sidecar_csv_path(row)
        annotations_df = _load_annotation_rows(csv_path)

        entries = []
        recording_duration = _recording_duration(raw)
        for row_data in annotations_df.itertuples(index=False):
            onset = float(row_data.start_time)
            stop_time = float(row_data.stop_time)
            if stop_time - onset <= _MIN_DURATION:
                continue
            entries.append(
                (
                    onset,
                    stop_time - onset,
                    encode_annotation_description(
                        str(row_data.label),
                        channel_count=int(row_data.channel_count),
                        source="tuar",
                    ),
                )
            )

        for onset, stop_time in _build_background_rows(annotations_df, recording_duration):
            entries.append(
                (
                    onset,
                    stop_time - onset,
                    encode_annotation_description("bckg", source="complement"),
                )
            )

        entries.sort(key=lambda item: (item[0], item[1], item[2]))
        if not entries:
            raw.set_annotations(mne.Annotations([], [], []))
            return raw

        onset, duration, description = zip(*entries)

        raw.set_annotations(
            mne.Annotations(onset=onset, duration=duration, description=description)
        )
        return raw


KERNEL = TuhEegArtifactKernel()
