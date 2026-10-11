"""Dataset-specific kernel for openneuro ds004106.

BCIT (Brain Cognition and Innovative Technology) dataset.
BCIT Advanced Guard Duty vigilance task.

4-digit BCIT event code structure:
  Digit 1: Task type (1=driving, 2=guard, 3=calibration)
  Digit 2: Condition/perturbation
  Digit 3: Event category (1=stimulus, 2=response, 3=performance)
  Digit 4: Specific sub-type
TODO: verify mapping against BCIT lab documentation.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from kernels.openneuro_specific._base import OpenNeuroBaseKernel

KERNEL = OpenNeuroBaseKernel(
    KERNEL_ID="openneuro-ds004106",
    event_map={
"1121": "target_card",
        "1122": "non_target_card",
        "1211": "correct_detection",
        "1212": "false_alarm",
        "1221": "correct_rejection",
        "1222": "miss",
        "1231": "response_button",
        "1232": "no_response",
    },
    continuous_cols=["response_time"],
)
