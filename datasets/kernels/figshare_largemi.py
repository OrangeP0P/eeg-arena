"""Figshare LargeMI stage1 kernel for EEGUnity.

Paradigm-prefixed label scheme
-------------------------------
All event descriptions are prefixed with their paradigm name to allow
per-paradigm distinction in the Event Structure table.

  CLA     — 3-class MI (left hand, right hand, passive state)
  HaLT    — 6-class MI (left hand, right hand, passive state,
             left leg, right leg, tongue)
  FREEFORM — 2-class free-imagery MI (left hand, right hand)
  5F      — 5-class finger MI (thumb, index, middle, ring, pinkie)
  NoMT    — passive-viewing control; participants watched cues without
             performing any motor imagery (per dataset paper §Paradigm #5).
             Codes 1-6 represent presented cues, not executed imagery.

Codes 91 / 92 / 99 (HaLT and NoMT only)
-----------------------------------------
These session-level marker codes appear in HaLT and NoMT files; their
precise meaning is not documented in the dataset paper. They are preserved
as admin: marker-XX labels for human review.
  91 — large continuous block (~240 s); likely baseline / rest segment.
  92 — very brief (~1 s); likely a transition signal.
  99 — periodic short bursts (~1 s); likely a fixation / ready signal.

Source: Kaya et al. (2018) https://doi.org/10.1038/sdata.2018.211
Label scheme added: 2026-04-08 — LLM-assisted, requires human review.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

import mne
import scipy.io as sio


# ---------------------------------------------------------------------------
# Subject demographics
# Source: Kaya et al. (2018) supplementary material
# Added: 2026-04-08, LLM-inferred, requires human review.
# ---------------------------------------------------------------------------
_SUBJECT_DICT: Dict[str, Dict[str, Any]] = {
    "A": {"sex": "Male",   "Age": "20-25"},
    "B": {"sex": "Male",   "Age": "20-25"},
    "C": {"sex": "Male",   "Age": "25-30"},
    "D": {"sex": "Male",   "Age": "25-30"},
    "E": {"sex": "Female", "Age": "20-25"},
    "F": {"sex": "Male",   "Age": "30-35"},
    "G": {"sex": "Male",   "Age": "30-35"},
    "H": {"sex": "Male",   "Age": "20-25"},
    "I": {"sex": "Female", "Age": "20-25"},
    "J": {"sex": "Female", "Age": "20-25"},
    "K": {"sex": "Male",   "Age": "20-25"},
    "L": {"sex": "Female", "Age": "20-25"},
    "M": {"sex": "Female", "Age": "20-25"},
}

# ---------------------------------------------------------------------------
# Per-paradigm event label maps  {marker_code: annotation_description}
# Naming convention: "<PARADIGM>: <label>"
# ---------------------------------------------------------------------------
_ADMIN_CODES: Dict[int, str] = {
    91: "admin: marker-91",
    92: "admin: marker-92",
    99: "admin: marker-99",
}

_PARADIGM_LABEL_MAP: Dict[str, Dict[int, str]] = {
    "CLA": {
        1: "CLA: left hand MI",
        2: "CLA: right hand MI",
        3: "CLA: passive state",
    },
    "FREEFORM": {
        1: "FREEFORM: left hand MI",
        2: "FREEFORM: right hand MI",
        3: "FREEFORM: passive state",
        99: "admin: marker-99",
    },
    "HaLT": {
        1: "HaLT: left hand MI",
        2: "HaLT: right hand MI",
        3: "HaLT: passive state",
        4: "HaLT: left leg MI",
        5: "HaLT: tongue MI",
        6: "HaLT: right leg MI",
        **_ADMIN_CODES,
    },
    "NoMT": {
        # Participants were passive (no MI executed); labels describe the
        # presented cue, not a motor task.
        1: "NoMT: left hand cue",
        2: "NoMT: right hand cue",
        3: "NoMT: passive cue",
        4: "NoMT: left leg cue",
        5: "NoMT: tongue cue",
        6: "NoMT: right leg cue",
        **_ADMIN_CODES,
    },
    "5F": {
        1: "5F: thumb MI",
        2: "5F: index finger MI",
        3: "5F: middle finger MI",
        4: "5F: ring finger MI",
        5: "5F: pinkie MI",
    },
}

# Detection order matters: check FREEFORM before any shorter prefix.
_PARADIGM_PREFIXES = ("5F", "FREEFORM", "CLA", "HaLT", "NoMT")


def _detect_paradigm(file_name: str) -> Optional[str]:
    """Return the paradigm key for *file_name*, or None if unrecognised."""
    upper = file_name.upper()
    for prefix in _PARADIGM_PREFIXES:
        if upper.startswith(prefix.upper()):
            return prefix
    return None


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

@dataclass
class FigshareLargeMIKernel:
    """Stage1 kernel for the figshare-largemi dataset."""

    KERNEL_ID: str = "figshare-largemi-stage1-v2"

    def apply(self, udataset, raw: mne.io.BaseRaw, row) -> mne.io.BaseRaw:
        file_path = row["File Path"]
        file_name = os.path.basename(file_path)

        # ── Paradigm detection ──────────────────────────────────────────────
        paradigm = _detect_paradigm(file_name)
        if paradigm is None:
            raise ValueError(
                f"Unrecognised paradigm prefix in file name: {file_name}"
            )
        event_dict = _PARADIGM_LABEL_MAP[paradigm]

        # ── Subject demographics ────────────────────────────────────────────
        subject_match = re.search(r"subject([A-Z])", file_name, flags=re.IGNORECASE)
        if subject_match is None:
            raise ValueError(f"Cannot infer subject id from file name: {file_name}")
        subject_key = subject_match.group(1).upper()
        if subject_key not in _SUBJECT_DICT:
            raise ValueError(f"Unknown subject id: {subject_key}")

        # ── Marker extraction ───────────────────────────────────────────────
        mat = sio.loadmat(file_path, simplify_cells=True)
        marker = mat["o"]["marker"]
        sfreq = raw.info["sfreq"]

        # Build annotations from contiguous marker segments.
        annotations: list = []
        current_code: Optional[int] = None
        current_start: int = 0

        for i, raw_code in enumerate(marker):
            code = int(raw_code)
            label = event_dict.get(code)          # None → unlabelled / zero

            active_label = event_dict.get(current_code) if current_code is not None else None

            if label is not None:
                if current_code is None:
                    current_code = code
                    current_start = i
                elif code != current_code:
                    # Flush previous segment
                    annotations.append((
                        current_start / sfreq,
                        (i - current_start) / sfreq,
                        active_label,
                    ))
                    current_code = code
                    current_start = i
            else:
                if current_code is not None:
                    annotations.append((
                        current_start / sfreq,
                        (i - current_start) / sfreq,
                        active_label,
                    ))
                    current_code = None

        # Flush final segment
        if current_code is not None:
            active_label = event_dict[current_code]
            annotations.append((
                current_start / sfreq,
                (len(marker) - current_start) / sfreq,
                active_label,
            ))

        if annotations:
            onset, duration, desc = zip(*annotations)
            raw.set_annotations(
                mne.Annotations(list(onset), list(duration), list(desc))
            )
        else:
            raw.set_annotations(mne.Annotations([], [], []))

        # ── Description (subject / device meta) ────────────────────────────
        description_dict = {
            "original_description": raw.info.get("description", ""),
            "eegunity_description": {
                "amplifier": "Nihon Kohden EEG-1200 device",
                "cap":       "Nihon Kohden EEG-1200 device",
                "age":       _SUBJECT_DICT[subject_key]["Age"],
                "sex":       _SUBJECT_DICT[subject_key]["sex"],
                "handedness": "unknown",
            },
        }
        raw.info["description"] = json.dumps(description_dict)

        return raw


KERNEL = FigshareLargeMIKernel()
