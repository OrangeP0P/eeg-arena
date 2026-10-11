"""Dataset-specific kernel for openneuro ds007176.

Current sidecars expose a minimal trial_type/value pair. Keep the raw labels
unchanged for now and reserve this file for later verified decoding.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from kernels.openneuro_specific._base import OpenNeuroBaseKernel


KERNEL = OpenNeuroBaseKernel(
    KERNEL_ID="openneuro-ds007176",
    event_map={},
    continuous_cols=[],
)
