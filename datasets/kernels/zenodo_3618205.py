"""Zenodo 3618205 - Selective Auditory Attention EEG Dataset stage1 kernel.

Dataset: https://zenodo.org/records/3618205
Fuglsang et al. (2020) — 44 subjects (22 normal-hearing, 22 hearing-impaired).
Three task types: selectiveattention, tonestimuli (MMN), rest.
Format: BIDS BDF files + companion _events.tsv sidecars.

Events TSV columns (tab-separated, with header):
  onset  duration  sample  type    value  trigger_type   attend_left_right

Only 'selectiveattention' task files have 'targetonset' trigger_type rows.
For those rows, 'attend_left_right' encodes the attended speaker:
  attendleft   → annotation 'attend_left'
  attendright  → annotation 'attend_right'

Annotation duration: each targetonset row has duration 'n/a' in the TSV.
This kernel sets duration = 0.0 (onset marker only) since true trial lengths
are not stored in the sidecar; downstream epoching should use long windows.

Non-selectiveattention files (rest, tonestimuli) receive no annotations.

Annotation labels produced (matching analysis.md Event Structure):
  'attend_left'  — Subject attends to the left-ear speaker
  'attend_right' — Subject attends to the right-ear speaker

Source: Fuglsang et al. (2020), BIDS dataset on Zenodo record 3618205
Kernel added: 2026-04-09 — LLM-assisted, requires human review.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import mne
import pandas as pd


_ATTEND_MAP = {
    "attendleft":  "attend_left",
    "attendright": "attend_right",
}


def _events_tsv_path(bdf_path: str) -> str:
    """Derive _events.tsv sidecar path from a _eeg.bdf file path."""
    base = bdf_path
    for suffix in ("_eeg.bdf", ".bdf"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base + "_events.tsv"


def _is_selective_attention_file(file_path: str) -> bool:
    """Return True only for selectiveattention task BDF files."""
    return "task-selectiveattention" in os.path.basename(file_path)


@dataclass
class Zenodo3618205Kernel:
    """Stage1 kernel for the zenodo_3618205 selective auditory attention dataset."""

    KERNEL_ID: str = "zenodo-3618205-stage1"

    def apply(self, udataset, raw: mne.io.BaseRaw, row) -> mne.io.BaseRaw:
        file_path = row["File Path"]

        description_dict = {
            "original_description": raw.info.get("description", ""),
            "eegunity_description": {
                "amplifier": "BioSemi ActiveTwo",
                "cap":       "64-channel + auxiliary, 10-20 system",
                "age":       "unknown",
                "sex":       "unknown",
                "handedness": "unknown",
            },
        }
        raw.info["description"] = json.dumps(description_dict)

        # Only inject annotations for selective attention task files.
        if not _is_selective_attention_file(file_path):
            return raw

        tsv_path = _events_tsv_path(file_path)
        if not os.path.exists(tsv_path):
            return raw

        try:
            df = pd.read_csv(tsv_path, sep="\t", dtype=str)
        except Exception:
            return raw

        # Keep only targetonset rows.
        if "trigger_type" not in df.columns:
            return raw

        target_rows = df[df["trigger_type"].str.strip().str.lower() == "targetonset"]
        if target_rows.empty:
            return raw

        onsets, durations, descriptions = [], [], []
        for _, ev_row in target_rows.iterrows():
            try:
                onset_s = float(ev_row["onset"])
            except (ValueError, KeyError):
                continue

            attend = str(ev_row.get("attend_left_right", "")).strip().lower()
            desc = _ATTEND_MAP.get(attend, attend if attend not in ("n/a", "", "nan") else "targetonset")

            onsets.append(onset_s)
            durations.append(0.0)
            descriptions.append(desc)

        if onsets:
            raw.set_annotations(
                mne.Annotations(
                    onset=onsets,
                    duration=durations,
                    description=descriptions,
                )
            )
        return raw


KERNEL = Zenodo3618205Kernel()
