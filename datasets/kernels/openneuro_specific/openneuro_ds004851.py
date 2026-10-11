"""Dataset-specific kernel for openneuro ds004851.

The public sidecars currently expose only numeric trigger values. Keep the raw
IDs intact until a verified decode table is added.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from kernels.openneuro_specific._base import OpenNeuroBaseKernel


KERNEL = OpenNeuroBaseKernel(
    KERNEL_ID="openneuro-ds004851",
    event_map={},
    continuous_cols=[],
)
