"""Kaggle INRIA (BCI Challenge @ NER 2015) stage1 kernel for EEGUnity.

Dataset: https://www.kaggle.com/c/inria-bci-challenge
Format: CSV files (one per session), 58 columns: Time + 56 EEG + EOG + FeedBackEvent.
Sampling rate: 200 Hz.

CSV quirk: each block within a session file starts with a repeated header row
  ("Time,Fp1,...,FeedBackEvent"). EEGUnity's CSV parser cannot determine
  Duration when these repeated headers are present → Completeness = Unavailable.

This kernel uses the extended interface (HANDLES_UNAVAILABLE = True):
  load(row)           — read CSV, strip repeated headers, build RawArray
  apply(ud, raw, row) — extract FeedBackEvent onset times, match TrainLabels.csv,
                        inject Annotations

Annotation labels produced:
  'good feedback to P300'    — correct BCI feedback (Prediction == 1)
  'wrong feedback to P300'   — incorrect BCI feedback (Prediction == 0)
  'unknown feedback to P300' — test sessions (no label in TrainLabels.csv)

Source: Margaux et al. (2012), Kaggle BCI Challenge @ NER 2015
Kernel added: 2026-04-09 — LLM-assisted, requires human review.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import mne
import numpy as np
import pandas as pd

from eegunity.utils.annotation_metadata import encode_annotation_description


_SFREQ = 200.0
_EEG_COLS_SOURCE = [
    "Fp1", "Fp2", "AF7", "AF3", "AF4", "AF8",
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "POz", "P08", "O1", "O2",
]
_EEG_COLS = ["PO8" if channel == "P08" else channel for channel in _EEG_COLS_SOURCE]
_EOG_COL = "EOG"
_STIM_COL = "FeedBackEvent"
_ANNOTATION_DURATION = 1.0  # seconds per feedback epoch


def _read_csv_clean(file_path: str) -> pd.DataFrame:
    """Read session CSV, stripping repeated header rows in the middle."""
    df = pd.read_csv(file_path, header=0, dtype=str, low_memory=False)
    # Drop rows where the first column equals the header string "Time"
    df = df[df.iloc[:, 0] != "Time"].reset_index(drop=True)
    return df.astype(float)


@dataclass
class KaggleInriaKernel:
    """Stage1 kernel for the kaggle-inria (BCI Challenge) dataset.

    Implements the extended kernel interface so that Unavailable CSV files
    (which have repeated header rows) are also processed.
    """

    KERNEL_ID: str = "kaggle-inria-stage1"
    HANDLES_UNAVAILABLE: bool = True

    def load(self, row) -> mne.io.BaseRaw | None:
        """Build a RawArray from a kaggle_inria session CSV file."""
        file_path = row["File Path"]
        if not file_path.lower().endswith(".csv"):
            return None
        # Skip non-session files (e.g. ChannelsLocation.csv, TrainLabels.csv)
        basename = os.path.basename(file_path)
        if not basename.startswith("Data_"):
            return None

        try:
            df = _read_csv_clean(file_path)
        except Exception:
            return None

        # Validate expected columns exist
        required = _EEG_COLS_SOURCE + [_EOG_COL, _STIM_COL]
        if not all(c in df.columns for c in required):
            return None

        eeg_data = df[_EEG_COLS_SOURCE].to_numpy(dtype=np.float64).T   # (56, n)
        eog_data = df[[_EOG_COL]].to_numpy(dtype=np.float64).T  # (1, n)
        stim_data = df[[_STIM_COL]].to_numpy(dtype=np.float64).T  # (1, n)

        ch_names = _EEG_COLS + [_EOG_COL, _STIM_COL]
        ch_types = ["eeg"] * len(_EEG_COLS) + ["eog", "stim"]

        data = np.concatenate([eeg_data, eog_data, stim_data], axis=0)
        info = mne.create_info(ch_names=ch_names, sfreq=_SFREQ, ch_types=ch_types)
        return mne.io.RawArray(data, info, verbose=False)

    def apply(self, udataset, raw: mne.io.BaseRaw, row) -> mne.io.BaseRaw:
        file_path = row["File Path"]

        description_dict = {
            "original_description": raw.info.get("description", ""),
            "eegunity_description": {
                "amplifier": "unknown",
                "cap": "Ag/AgCl",
                "age": "20-37",
                "sex": "unknown",
                "handedness": "unknown",
            },
        }
        raw.info["description"] = json.dumps(description_dict)

        basename = os.path.basename(file_path)
        if not basename.startswith("Data_"):
            return raw

        # Locate TrainLabels.csv in the same directory as the session file.
        dataset_dir = os.path.dirname(file_path)
        train_labels_path = os.path.join(dataset_dir, "TrainLabels.csv")

        train_labels_df = None
        if os.path.exists(train_labels_path):
            try:
                train_labels_df = pd.read_csv(train_labels_path)
            except Exception:
                pass

        # base_name e.g. "Data_S02_Sess01" → session prefix "S02_Sess01"
        base_name = os.path.splitext(basename)[0]
        session_prefix = base_name[5:]  # strip "Data_"

        # Extract FeedBackEvent onset samples from stim channel.
        stim_data = raw.get_data(picks=["FeedBackEvent"])[0]
        event_indices = np.where(stim_data == 1)[0]
        event_times = event_indices / raw.info["sfreq"]

        descriptions = []
        for i, _ in enumerate(event_times):
            feedback_id = f"{session_prefix}_FB{str(i + 1).zfill(3)}"
            label = "unknown feedback to P300"
            if train_labels_df is not None:
                label_row = train_labels_df[
                    train_labels_df["IdFeedBack"] == feedback_id
                ]
                if not label_row.empty:
                    prediction = label_row["Prediction"].values[0]
                    label = (
                        "good feedback to P300"
                        if prediction == 1
                        else "wrong feedback to P300"
                    )
            descriptions.append(
                encode_annotation_description(
                    label,
                    feedback_id=feedback_id,
                    feedback_index=i + 1,
                    session_id=session_prefix,
                )
            )

        if event_times.size > 0:
            raw.set_annotations(
                mne.Annotations(
                    onset=event_times,
                    duration=np.full(len(event_times), _ANNOTATION_DURATION),
                    description=descriptions,
                )
            )
        return raw


KERNEL = KaggleInriaKernel()
