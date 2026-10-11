"""Other HighGammaDataset stage1 kernel for EEGUnity."""

from __future__ import annotations

import json
from dataclasses import dataclass

import mne


@dataclass
class OtherHighGammaDatasetKernel:
    """Stage1 kernel for the other-highgammadataset dataset."""

    KERNEL_ID: str = "other-highgammadataset-stage1"

    def apply(self, udataset, raw: mne.io.BaseRaw, row) -> mne.io.BaseRaw:
        description_dict = {
            "original_description": raw.info.get("description", ""),
            "eegunity_description": {
                "amplifier": "NeurOne, Mega Electronics Ltd, Kuopio, FI",
                "cap": "WaveGuard Original, ANT, Enschede, NL",
                "age": "unknown",
                "sex": "unknown",
                "handedness": "unknown",
            },
        }
        raw.info["description"] = json.dumps(description_dict)
        return raw


KERNEL = OtherHighGammaDatasetKernel()
