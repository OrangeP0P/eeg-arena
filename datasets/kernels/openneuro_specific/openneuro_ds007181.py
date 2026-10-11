"""Dataset-specific kernel for openneuro ds007181.

Sleep-stage annotations live in events.tsv under the `stage` column rather than
the usual BIDS label columns. Rebuild long annotations from that column so the
locator can reflect 30-second PSG staging directly.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from dataclasses import dataclass

import mne
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from kernels.bids import _find_events_tsv
from kernels.openneuro_specific._base import OpenNeuroBaseKernel

STAGE_MAP = {
    "W": "Sleep stage W",
    "N1": "Sleep stage N1",
    "N2": "Sleep stage N2",
    "N3": "Sleep stage N3",
    "R": "Sleep stage R (REM)",
    "REM": "Sleep stage R (REM)",
}


def _summarize_annotations(raw: mne.io.BaseRaw) -> str:
    counts = Counter(str(desc) for desc in raw.annotations.description if str(desc).strip())
    if not counts:
        return "no_events"
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ", ".join(f"{label}: {count}" for label, count in ordered)


@dataclass
class OpenNeuroDS007181Kernel(OpenNeuroBaseKernel):
    KERNEL_ID: str = "openneuro-ds007181"

    def apply(self, udataset, raw: mne.io.BaseRaw, row) -> mne.io.BaseRaw:
        raw = super().apply(udataset, raw, row)

        events_path = _find_events_tsv(row["File Path"])
        if events_path is None or not os.path.exists(events_path):
            return raw

        try:
            events_df = pd.read_csv(events_path, sep="\t")
        except Exception:
            return raw

        required = {"onset", "duration", "stage"}
        if not required.issubset(events_df.columns):
            return raw

        onsets = []
        durations = []
        descriptions = []
        eps = 1.0 / float(raw.info["sfreq"])

        for _, event_row in events_df.iterrows():
            onset = event_row.get("onset")
            duration = event_row.get("duration")
            stage = str(event_row.get("stage", "")).strip()
            if pd.isna(onset) or pd.isna(duration) or not stage:
                continue
            if stage.lower() in {"n/a", "na", "nan"}:
                continue
            label = STAGE_MAP.get(stage, stage)
            try:
                duration_value = float(duration)
            except Exception:
                continue
            onsets.append(float(onset))
            durations.append(duration_value if duration_value > 0 else eps)
            descriptions.append(label)

        if not descriptions:
            return raw

        raw.set_annotations(mne.Annotations(onset=onsets, duration=durations, description=descriptions))

        try:
            description_dict = json.loads(raw.info.get("description", "") or "{}")
        except Exception:
            description_dict = {}
        eegunity_description = description_dict.setdefault("eegunity_description", {})
        eegunity_description["events"] = _summarize_annotations(raw)
        raw.info["description"] = json.dumps(description_dict)
        return raw


KERNEL = OpenNeuroDS007181Kernel(event_map={}, continuous_cols=[])
