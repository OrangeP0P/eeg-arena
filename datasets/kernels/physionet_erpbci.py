"""Physionet ERP-BCI stage1 kernel for EEGUnity."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import mne
import numpy as np
from eegunity.modules.parser.eeg_parser import extract_events


def _process_p300_events(events, event_id):
    """Process P300 events: classify as seen/unseen row/column based on target."""
    rows = {"ABCDEF", "MNOPQR", "STUVWX", "YZ1234", "56789_"}
    columns = {"AGMSY5", "BHNTZ6", "CIOU17", "DJPV28", "EKQW39", "FLRX4_"}

    target_event_key = [key for key in event_id if key.startswith("#Tgt")][0]
    target_char = target_event_key[4]

    event_id.update(
        {
            "seen (row)": 30,
            "seen (column)": 31,
            "unseen (row)": 32,
            "unseen (column)": 33,
        }
    )

    new_events = []
    for event in events:
        onset, previous, event_code = event
        event_label = [key for key, value in event_id.items() if value == event_code]
        if not event_label:
            continue
        event_label = event_label[0]

        if event_label in rows:
            if target_char in event_label:
                new_event = [onset, previous, event_id["seen (row)"]]
            else:
                new_event = [onset, previous, event_id["unseen (row)"]]
        elif event_label in columns:
            if target_char in event_label:
                new_event = [onset, previous, event_id["seen (column)"]]
            else:
                new_event = [onset, previous, event_id["unseen (column)"]]
        else:
            continue

        new_events.append(new_event)

    return np.array(new_events), event_id


@dataclass
class PhysionetErpbciKernel:
    """Stage1 kernel for the physionet-erpbci dataset."""

    KERNEL_ID: str = "physionet-erpbci-stage1"

    def apply(self, udataset, raw: mne.io.BaseRaw, row) -> mne.io.BaseRaw:
        description_dict = {
            "original_description": raw.info.get("description", ""),
            "eegunity_description": {
                "amplifier": "BioSemi ActiveTwo",
                "cap": "unknown",
                "age": "unknown",
                "sex": "unknown",
                "handedness": "unknown",
            },
        }
        raw.info["description"] = json.dumps(description_dict)

        # Extract events and process P300 row/column classification.
        events, event_id = extract_events(raw)
        events, event_id = _process_p300_events(events, event_id)
        event_descriptions = {v: k for k, v in event_id.items()}

        annotations = mne.annotations_from_events(
            events=events, sfreq=raw.info["sfreq"], event_desc=event_descriptions
        )
        raw.set_annotations(annotations)

        return raw


KERNEL = PhysionetErpbciKernel()
