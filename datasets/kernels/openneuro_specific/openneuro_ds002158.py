"""Dataset-specific kernel for openneuro ds002158.

EEG task with RT and confidence.
Generic BIDS kernel + response_time (and confidence where available)
injected as misc channels for regression-task construction.
Event labels from events.tsv are already human-readable; no recoding needed.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from kernels.openneuro_specific._base import OpenNeuroBaseKernel

KERNEL = OpenNeuroBaseKernel(
    KERNEL_ID="openneuro-ds002158",
    event_map={},
    continuous_cols=['confidence', 'response_time'],
)
