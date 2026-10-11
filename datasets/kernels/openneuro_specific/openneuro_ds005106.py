"""Dataset-specific kernel for openneuro ds005106.

Rapid image-presentation EEG. events.tsv exposes trial and stimulus indices
that are useful as misc channels even when the final task label decode is still
under review.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from kernels.openneuro_specific._base import OpenNeuroBaseKernel


KERNEL = OpenNeuroBaseKernel(
    KERNEL_ID="openneuro-ds005106",
    event_map={},
    continuous_cols=["trialnr", "blocknr", "presentationnumber", "stimnum"],
)
