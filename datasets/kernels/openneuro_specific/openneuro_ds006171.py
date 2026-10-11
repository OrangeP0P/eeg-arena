"""Dataset-specific kernel for openneuro ds006171.

EEG during near-threshold visual detection tasks (no-cue, 50%-cue, 100%-cue).
Events.tsv contains gabor_contrast, subjective_r (awareness rating 0-1),
objective_r (accuracy) — these go into misc channels for regression tasks.

Event code mapping (decoded from description + events.tsv structure):
  1,2 = trial type (detected / not detected)
  98,99 = catch trials
  101-128 = condition+response compound codes
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from kernels.openneuro_specific._base import OpenNeuroBaseKernel

KERNEL = OpenNeuroBaseKernel(
    KERNEL_ID="openneuro-ds006171",
    event_map={
# Condition codes (1st digit = cue type: 1=no-cue, 2=50%-valid, 3=100%-valid)
        # Derived from dataset description; verify with README
        "1":   "trial_start",
        "2":   "response",
        "98":  "catch_trial",
        "99":  "catch_response",
        "101": "nocue_hit",
        "102": "nocue_miss",
        "103": "nocue_FA",
        "104": "nocue_CR",
        "111": "50pct_hit",
        "112": "50pct_miss",
        "113": "50pct_FA",
        "114": "50pct_CR",
        "121": "100pct_hit",
        "122": "100pct_miss",
        "124": "100pct_FA",
        "125": "100pct_CR",
        "128": "boundary",
    },
    continuous_cols=[
        "response_time",
        "gabor_contrast",    # contrast level of the Gabor patch (0–1)
        "subjective_r",      # subjective detection report (0/1)
        "objective_r",       # objective task accuracy (0/1)
    ],
)
