"""Dataset-specific kernel for openneuro ds004446.

The BMI-HDEEG dataset 2

Sensorimotor rhythm BCI: 3-class MI
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from kernels.openneuro_specific._base import OpenNeuroBaseKernel

KERNEL = OpenNeuroBaseKernel(
    KERNEL_ID="openneuro-ds004446",
    event_map={
"1": "left_hand_MI",
    "2": "right_hand_MI",
    "3": "feet_MI",
    },
    continuous_cols=['response_time'],
)
