"""Dataset-specific kernel for openneuro ds004447.

The BMI-HDEEG dataset 3

Sensorimotor rhythm BCI: 4-class MI
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from kernels.openneuro_specific._base import OpenNeuroBaseKernel

KERNEL = OpenNeuroBaseKernel(
    KERNEL_ID="openneuro-ds004447",
    event_map={
"1": "left_hand_MI",
    "2": "right_hand_MI",
    "3": "feet_MI",
    "4": "tongue_MI",
    },
    continuous_cols=['response_time'],
)
