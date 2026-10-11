"""Dataset-specific kernel for openneuro ds002720.

BCI Tempo-based BCMI

Motor imagery (ball squeezing) vs relaxation BCI
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from kernels.openneuro_specific._base import OpenNeuroBaseKernel

KERNEL = OpenNeuroBaseKernel(
    KERNEL_ID="openneuro-ds002720",
    event_map={
"1": "motor_imagery",
    "2": "relaxation",
    },
    continuous_cols=['response_time'],
)
