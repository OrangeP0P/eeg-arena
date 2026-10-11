"""Dataset-specific kernel for openneuro ds002721.

OpenNeuro ds002721 currently exposes sparse events.tsv markers with a numeric
trial_type code. Keep the dataset on the OpenNeuro base path so future mapping
work can extend this file without changing the dispatcher.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from kernels.openneuro_specific._base import OpenNeuroBaseKernel


KERNEL = OpenNeuroBaseKernel(
    KERNEL_ID="openneuro-ds002721",
    event_map={},
    continuous_cols=[],
)
