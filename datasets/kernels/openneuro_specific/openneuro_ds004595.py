"""Dataset-specific kernel for openneuro ds004595.

Three-armed bandit EEG with rich trial_type labels already present in
events.tsv. Keep the trial labels as-is and expose response_time in misc.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from kernels.openneuro_specific._base import OpenNeuroBaseKernel


KERNEL = OpenNeuroBaseKernel(
    KERNEL_ID="openneuro-ds004595",
    event_map={},
    continuous_cols=["response_time"],
)
