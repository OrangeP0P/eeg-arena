"""Dataset-specific kernel for openneuro ds003421.

HD-EEG multi-condition (Dataset 2): resting state, visual and auditory object naming.
events.tsv has: value (0/1), stimuli (audio filename or image name), response_time.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from kernels.openneuro_specific._base import OpenNeuroBaseKernel

KERNEL = OpenNeuroBaseKernel(
    KERNEL_ID="openneuro-ds003421",
    event_map={
"0": "stimulus_onset",
        "1": "response",
    },
    continuous_cols=["response_time"],
)
