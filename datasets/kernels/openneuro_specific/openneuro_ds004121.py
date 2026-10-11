"""Dataset-specific kernel for openneuro ds004121.

BCIT (Brain Cognition and Innovative Technology) dataset.
BCIT Mind Wandering: driving with mind-wandering probes.

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
    KERNEL_ID="openneuro-ds004121",
    event_map={
"1111": "drive_segment_start",
        "1112": "drive_segment_end",
        "1121": "lane_perturb_left",
        "1122": "lane_perturb_right",
        "1211": "steer_left",
        "1212": "steer_right",
        "1341": "mind_wandering_probe",
        "1342": "probe_response",
    },
    continuous_cols=["response_time"],
)
