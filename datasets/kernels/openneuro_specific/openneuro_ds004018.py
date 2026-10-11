"""Dataset-specific kernel for openneuro ds004018.

RSVP object-image EEG. Raw annotations already expose the coarse condition
(`stars` / `boats`), while events.tsv provides additional per-trial metadata
such as target flags and reaction time.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from kernels.openneuro_specific._base import OpenNeuroBaseKernel


KERNEL = OpenNeuroBaseKernel(
    KERNEL_ID="openneuro-ds004018",
    event_map={},
    continuous_cols=["rt"],
)
