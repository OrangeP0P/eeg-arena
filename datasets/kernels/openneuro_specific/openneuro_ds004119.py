"""Dataset-specific kernel for openneuro ds004119.

BCIT (Brain Cognition and Innovative Technology) dataset.
BCIT Basic Guard Duty: sustained vigilance card verification.

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
    KERNEL_ID="openneuro-ds004119",
    event_map={
"1111": "trial_start",
        "1231": "correct_accept",
        "1232": "correct_reject",
        "1311": "false_alarm",
        "1312": "miss",
        "1321": "response_onset",
        "1322": "response_offset",
    },
    continuous_cols=["response_time"],
)
