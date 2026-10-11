"""Dataset-specific kernel for openneuro ds004367.

Meta-RDK: Raw EEG from visual motion discrimination + metacognitive task.
Value codes are BioSemi STATUS-type compound codes.
Rough decoding (TODO: verify from paper DOI 10.1101/2023.04.06.535954):
  Leftward motion decision codes vs rightward + confidence rating codes.
  High values (3000+) = response/rating events.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from kernels.openneuro_specific._base import OpenNeuroBaseKernel

KERNEL = OpenNeuroBaseKernel(
    KERNEL_ID="openneuro-ds004367",
    event_map={
"1536": "motion_stimulus_onset",
        "3072": "left_response",
        "3328": "right_response",
        "3456": "confidence_low",
        "3584": "confidence_medium",
        "3840": "confidence_high",
        "3968": "feedback_onset",
        "4096": "trial_end",
    },
    continuous_cols=["response_time", "confidence"],
)
