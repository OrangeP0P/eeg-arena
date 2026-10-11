"""Dataset-specific kernel for openneuro ds002778.

UC San Diego Resting State EEG — Parkinson's disease vs healthy controls.
Recorded with BioSemi. Only status code 65536 (CM baseline) present —
this is a resting-state dataset with no meaningful trial markers.
Kernel adds participant group (PD vs HC) from participants.tsv as misc annotation.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from kernels.openneuro_specific._base import OpenNeuroBaseKernel

KERNEL = OpenNeuroBaseKernel(
    KERNEL_ID="openneuro-ds002778",
    event_map={},
    continuous_cols=["response_time"],
)
