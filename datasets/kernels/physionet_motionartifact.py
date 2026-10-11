"""Physionet EEG/fNIRS Motion Artifact Dataset stage1 kernel for EEGUnity.

File format (TrialX.csv, no header row, 11 columns, 2048 Hz EEG):
  Col 1  : sample index
  Col 2  : EEG channel 1 (pre-frontal, raw)
  Col 3  : EEG channel 2 (pre-frontal, reference)
  Col 4  : EEG trigger (2048 Hz) — HIGH (1) = normal, LOW (0) = artifact present
  Col 5-7: Accelerometer 1 X/Y/Z (200 Hz, stored at 2048 Hz density)
  Col 8-10: Accelerometer 2 X/Y/Z (200 Hz)
  Col 11 : Accelerometer trigger (200 Hz)

Trigger interpretation (from ANNOTATORS file):
  Falling edge (1→0): start of motion artifact period → annotation "F"
  Rising edge  (0→1): end   of motion artifact period → annotation "R"

Each trial has exactly one F–R pair bracketing the artifact interval.

This kernel uses the **extended kernel interface** (HANDLES_UNAVAILABLE = True):
EEGUnity's default CSV parser requires a time column and cannot parse these
headerless files (Completeness Check = Unavailable). The kernel implements
``load(row)`` to build a minimal MNE RawArray from columns 2–3 (EEG channels)
at 2048 Hz, so that ``apply()`` can then inject the trigger annotations.

  load(row)  → reads CSV columns 1-2 (EEG data), builds RawArray at 2048 Hz
  apply(...) → detects F/R edges in the trigger column, sets annotations

Source: https://physionet.org/content/motion-artifact/1.0.0/
Kernel added: 2026-04-09 — LLM-assisted, requires human review.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import mne
import numpy as np
import pandas as pd


_SFREQ = 2048.0  # EEG sampling rate
_EEG_COL_INDICES = [1, 2]     # 0-based: col 2 and 3 (pre-frontal raw + reference)
_TRIGGER_COL_INDEX = 3         # 0-based: col 4 (EEG trigger)
_CH_NAMES = ["EEG1", "EEG2"]  # pre-frontal raw, pre-frontal reference


def _extract_trigger_annotations(file_path: str, sfreq: float) -> mne.Annotations:
    """Read CSV, extract trigger column, detect F/R edges, return Annotations."""
    df = pd.read_csv(file_path, header=None, usecols=[_TRIGGER_COL_INDEX])
    trigger = df.iloc[:, 0].to_numpy(dtype=int)

    prev = trigger[:-1]
    curr = trigger[1:]

    fall_indices = np.where((prev == 1) & (curr == 0))[0] + 1  # 1→0: start artifact
    rise_indices = np.where((prev == 0) & (curr == 1))[0] + 1  # 0→1: end artifact

    onsets, durations, descs = [], [], []
    for idx in fall_indices:
        onsets.append(idx / sfreq)
        durations.append(0.0)
        descs.append("F")
    for idx in rise_indices:
        onsets.append(idx / sfreq)
        durations.append(0.0)
        descs.append("R")

    return mne.Annotations(onset=onsets, duration=durations, description=descs)


@dataclass
class PhysionetMotionArtifactKernel:
    """Stage1 kernel for the physionet-motionartifact dataset.

    Implements the extended kernel interface:
      HANDLES_UNAVAILABLE = True   — opt in to process Unavailable files
      load(row)                    — build raw from headerless CSV
      apply(udataset, raw, row)    — inject F/R trigger annotations
    """

    KERNEL_ID: str = "physionet-motionartifact-stage1"
    HANDLES_UNAVAILABLE: bool = True

    def load(self, row) -> mne.io.BaseRaw | None:
        """Build a RawArray from the EEG columns of a headerless CSV file.

        Called by EEGUnity for Unavailable files before ``apply()``.
        Returns ``None`` for non-CSV files so they are skipped.
        """
        file_path = row["File Path"]
        if not file_path.endswith(".csv"):
            return None

        try:
            df = pd.read_csv(file_path, header=None, usecols=_EEG_COL_INDICES)
        except Exception:
            return None

        eeg = df.to_numpy(dtype=np.float64).T  # (2, n_samples)
        info = mne.create_info(ch_names=_CH_NAMES, sfreq=_SFREQ, ch_types=["eeg", "eeg"])
        return mne.io.RawArray(eeg, info, verbose=False)

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

        if not file_path.endswith(".csv"):
            return raw

        sfreq = raw.info.get("sfreq") or _SFREQ
        annotations = _extract_trigger_annotations(file_path, sfreq)
        raw.set_annotations(annotations)
        return raw


KERNEL = PhysionetMotionArtifactKernel()
