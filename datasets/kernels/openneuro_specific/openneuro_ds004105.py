"""Dataset-specific kernel for openneuro ds004105.

BCIT (Brain Cognition and Innovative Technology) dataset.
BCIT Auditory Cueing driving task.

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
    KERNEL_ID="openneuro-ds004105",
    event_map={
"1131": "lane_perturb_left",
        "1132": "lane_perturb_right",
        "1141": "auditory_cue_left",
        "1142": "auditory_cue_right",
        "1211": "correct_steer_left",
        "1212": "correct_steer_right",
        "1321": "high_workload_onset",
        "1322": "low_workload_onset",
    },
    continuous_cols=["response_time"],
)
