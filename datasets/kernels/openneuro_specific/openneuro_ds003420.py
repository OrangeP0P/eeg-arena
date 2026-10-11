"""Dataset-specific kernel for openneuro ds003420.

HD-EEG visual naming task (Dataset 1).
Subjects name visually presented objects (pictured items, 2AFC task).
events.tsv has: value (0/1 = stimulus/response), stimuli (object name), response_time.
This kernel:
  - Recodes value → "stimulus" / "response"
  - Injects response_time as misc channel
  - Injects stimuli name via annotation (TODO: add as misc categorical if needed)
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from kernels.openneuro_specific._base import OpenNeuroBaseKernel

KERNEL = OpenNeuroBaseKernel(
    KERNEL_ID="openneuro-ds003420",
    event_map={
"0": "stimulus_onset",
        "1": "response",
    },
    continuous_cols=["response_time"],
)
