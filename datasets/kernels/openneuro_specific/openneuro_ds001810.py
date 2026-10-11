"""Dataset-specific kernel for openneuro ds001810.

EEG study of the attentional blink (tDCS conditions).
Recorded with BioSemi. events.tsv type=STATUS, values are BioSemi trigger codes.

BioSemi STATUS codes for attentional blink RSVP paradigm (decoded from description):
  T1 target present/absent × T2 target present/absent × lag condition.
  Common mapping based on RSVP paradigm conventions:
    61450 = T1 + T2 target (lag 3)
    61463 = T1 target, T2 absent (lag 3)
    61471 = T1 absent, T2 target
    61472 = T1 T2 at lag 8 (outside AB window)
    61480 = T1 onset (general)
    61490 = T2 onset (general)
  TODO: verify exact mapping from original paper/stimulation protocol.
  Reference: doi.org/10.1111/j.1469-8986.2010.01081.x
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from kernels.openneuro_specific._base import OpenNeuroBaseKernel

KERNEL = OpenNeuroBaseKernel(
    KERNEL_ID="openneuro-ds001810",
    event_map={
# BioSemi STATUS codes; lower 16-bit difference from 61440 (CM_in_range base)
        "61440": "CM_baseline",
        "61450": "T1_present_T2_present_lag3",   # TODO verify
        "61463": "T1_present_T2_absent_lag3",    # TODO verify
        "61468": "stimulus_onset",
        "61471": "T1_absent_T2_present",         # TODO verify
        "61472": "T1_T2_lag8",                   # outside AB window
        "61480": "T1_onset",
        "61490": "T2_onset",
        "61694": "CM_out_of_range",
    },
    continuous_cols=["response_time"],
)
