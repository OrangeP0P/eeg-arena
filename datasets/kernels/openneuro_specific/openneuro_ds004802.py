"""Dataset-specific kernel for openneuro ds004802.

Roving oddball EEG. events.tsv provides a human-readable `event` column in
addition to numeric trigger values. Rebuild annotations from that column so the
locator reflects task semantics instead of opaque numeric codes.
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


def _normalize_event_label(label: str) -> str:
    text = str(label).strip()
    if not text:
        return "unknown_event"
    return text.replace("/", "_").replace(" ", "_")


def _summarize_annotations(raw: mne.io.BaseRaw) -> str:
    counts = Counter(str(desc) for desc in raw.annotations.description if str(desc).strip())
    if not counts:
        return "no_events"
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ", ".join(f"{label}: {count}" for label, count in ordered)


@dataclass
class OpenNeuroDS004802Kernel(OpenNeuroBaseKernel):
    KERNEL_ID: str = "openneuro-ds004802"

    def apply(self, udataset, raw: mne.io.BaseRaw, row) -> mne.io.BaseRaw:
        raw = super().apply(udataset, raw, row)

        events_path = _find_events_tsv(row["File Path"])
        if events_path is None or not os.path.exists(events_path):
            return raw

        try:
            events_df = pd.read_csv(events_path, sep="\t")
        except Exception:
            return raw

        required = {"onset", "event"}
        if not required.issubset(events_df.columns):
            return raw

        onsets = []
        durations = []
        descriptions = []
        eps = 1.0 / float(raw.info["sfreq"])

        for _, event_row in events_df.iterrows():
            onset = event_row.get("onset")
            event = event_row.get("event")
            if pd.isna(onset) or pd.isna(event):
                continue
            label = _normalize_event_label(event)
            if label.lower() in {"n_a", "na"}:
                continue
            duration = event_row.get("duration", 0.0)
            try:
                duration_value = float(duration)
            except Exception:
                duration_value = 0.0
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


KERNEL = OpenNeuroDS004802Kernel(event_map={}, continuous_cols=[])
