"""TUH EEG Seizure Corpus stage1 kernel for EEGUnity.

Dataset: https://isip.piconepress.com/projects/tuh_eeg/downloads/tuh_eeg_seizure/v2.0.3/
Format: EDF files with companion CSV sidecar annotation files.

CSV sidecar format (header lines prefixed with '#', then a header row):
  # version = csv_v1.0.0
  # bname = aaaaaajy_s001_t000
  # duration = 1750.00 secs
  channel,start_time,stop_time,label,confidence
  FP1-F7,57.9964,196.0107,gnsz,1.0000
  FP1-F7,57.9964,196.0107,gnsz,1.0000
  ...

Multiple rows may share the same (start_time, stop_time, label) triple because the
same seizure interval is annotated per-channel. This kernel deduplicates by
unique (start_time, stop_time, label) so each interval appears once.

Annotation labels produced (matching analysis.md Event Structure):
  'bckg'  — Background, no seizure
  'gnsz'  — Generalized non-specific seizure
  'fnsz'  — Focal non-specific seizure
  'cpsz'  — Complex partial seizure
  'absz'  — Absence seizure
  'tnsz'  — Tonic seizure
  (other labels passed through as-is)

Train/dev split is encoded in the file path and NOT annotated by this kernel.
Duration of each annotation = stop_time - start_time (from the CSV sidecar).

Source: Shah et al. (2020), v2.0.3
Kernel added: 2026-04-09 — LLM-assisted, requires human review.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass

import mne

from eegunity.utils.annotation_metadata import encode_annotation_description


def _companion_csv_path(edf_path: str) -> str:
    """Return the expected .csv companion path for an EDF file."""
    base, _ = os.path.splitext(edf_path)
    return base + ".csv"


def _parse_tuh_csv(csv_path: str) -> list[tuple[float, float, str]]:
    """Parse TUH seizure CSV sidecar, return deduplicated (onset, duration, label)."""
    seen: set[tuple[float, float, str]] = set()
    result: list[tuple[float, float, str]] = []

    with open(csv_path, "r", newline="") as fh:
        # Skip comment lines that begin with '#'
        lines = [ln for ln in fh if not ln.startswith("#")]

    reader = csv.DictReader(lines)
    for row in reader:
        try:
            start = float(row["start_time"])
            stop = float(row["stop_time"])
            label = row["label"].strip()
        except (KeyError, ValueError):
            continue

        duration = stop - start
        if duration <= 0:
            continue

        key = (start, stop, label)
        if key not in seen:
            seen.add(key)
            result.append((start, duration, label))

    return result


def _infer_split(file_path: str) -> str:
    path_parts = file_path.replace("\\", "/").split("/")
    for split in ("train", "dev", "eval"):
        if split in path_parts:
            return split
    return "unknown"


@dataclass
class TuhEegSeizureKernel:
    """Stage1 kernel for the TUH EEG Seizure Corpus."""

    KERNEL_ID: str = "tuh-eeg-seizure-stage1"

    def apply(self, udataset, raw: mne.io.BaseRaw, row) -> mne.io.BaseRaw:
        file_path = row["File Path"]

        description_dict = {
            "original_description": raw.info.get("description", ""),
            "eegunity_description": {
                "amplifier": "unknown",
                "cap":       "unknown",
                "age":       "unknown",
                "sex":       "unknown",
                "handedness": "unknown",
            },
        }
        raw.info["description"] = json.dumps(description_dict)

        # Only process EDF files; skip non-EDF files in the dataset tree.
        if not file_path.lower().endswith(".edf"):
            return raw

        csv_path = _companion_csv_path(file_path)
        if not os.path.exists(csv_path):
            return raw

        try:
            intervals = _parse_tuh_csv(csv_path)
        except Exception:
            return raw

        if not intervals:
            return raw

        split = _infer_split(str(file_path))
        onsets, durations, labels = zip(*intervals)
        descriptions = [
            encode_annotation_description(label, source="tusz", split=split)
            for label in labels
        ]
        raw.set_annotations(
            mne.Annotations(
                onset=list(onsets),
                duration=list(durations),
                description=descriptions,
            )
        )
        return raw


KERNEL = TuhEegSeizureKernel()
