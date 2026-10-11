"""Figshare ShuDB stage1 kernel for EEGUnity."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import mne
import pandas as pd


def _read_tsv(file_path):
    """Read a TSV file, auto-detecting whether the first line is a header."""
    with open(file_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip().split("\t")

    if first_line[0].lower() == "onset":
        df = pd.read_csv(file_path, sep="\t", header=0, skipinitialspace=True)
    else:
        df = pd.read_csv(
            file_path,
            sep="\t",
            header=None,
            names=["onset", "duration", "trial_type", "response_time", "sample", "value"],
            skipinitialspace=True,
        )
    return df


def _resolve_events_file(file_path: str, dataset_path: str | None) -> Path:
    """Locate the TSV sidecar for one SHUDB EDF file.

    The raw dump may contain EDFs directly under the dataset root or under an
    ``edf/`` subdirectory, while the event TSVs live in the dataset root.
    """
    file_path = Path(file_path)
    events_name = file_path.name.replace("eeg.edf", "events.tsv")

    candidate_dirs = [file_path.parent]
    candidate_dirs.extend(file_path.parents)
    if dataset_path:
        candidate_dirs.append(Path(dataset_path))
        candidate_dirs.append(Path(dataset_path) / "events")

    seen = set()
    for directory in candidate_dirs:
        if directory is None:
            continue
        directory = Path(directory)
        if directory in seen:
            continue
        seen.add(directory)
        candidate = directory / events_name
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(f"Could not locate events TSV for {file_path}")


@dataclass
class FigshareShudbKernel:
    """Stage1 kernel for the figshare-shudb dataset."""

    KERNEL_ID: str = "figshare-shudb-stage1"

    def apply(self, udataset, raw: mne.io.BaseRaw, row) -> mne.io.BaseRaw:
        file_path = row["File Path"]
        shared = udataset.get_shared_attr()
        dataset_path = shared.get("dataset_path") if shared else None
        events_file_path = _resolve_events_file(file_path, dataset_path)

        events_df = _read_tsv(events_file_path)
        events_df["onset"] = pd.to_numeric(events_df["onset"], errors="coerce")
        events_df["duration"] = pd.to_numeric(events_df["duration"], errors="coerce")
        events_df["trial_type"] = events_df["trial_type"].astype(str).str.strip()
        events_df = events_df.dropna(subset=["onset", "duration", "trial_type"])

        start_time = (events_df["onset"].to_numpy() - 1.0) / raw.info["sfreq"]
        duration = events_df["duration"].to_numpy() / raw.info["sfreq"]
        label = events_df["trial_type"].values

        annotations = mne.Annotations(onset=start_time, duration=duration, description=label)
        raw.set_annotations(annotations)

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

        return raw


KERNEL = FigshareShudbKernel()
