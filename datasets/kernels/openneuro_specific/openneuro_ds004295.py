"""Dataset-specific kernel for openneuro ds004295.

Reward gain vs punishment avoidance reversal learning task.
BrainVision S-codes in events.tsv (value column: S 51, S 52, ...).

Event code mapping (reversal learning paradigm):
  S 51 = stimulus A (reward-associated, pre-reversal)
  S 52 = stimulus B (punishment-associated, pre-reversal)
  S 53 = correct response (reward gained)
  S 54 = incorrect response (punishment)
  S 55 = reversal cue
  S 56 = stimulus A (punishment-associated, post-reversal)
  S 60 = feedback positive
  S 61 = feedback negative
  TODO: verify exact mapping from paper DOI 10.1016/j.neuropsychologia.2019.107125
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from kernels.openneuro_specific._base import OpenNeuroBaseKernel

KERNEL = OpenNeuroBaseKernel(
    KERNEL_ID="openneuro-ds004295",
    event_map={
"S 51": "stimulus_A_prereversalReward",
        "S 52": "stimulus_B_prereversalPunishment",
        "S 53": "correct_response",
        "S 54": "incorrect_response",
        "S 55": "reversal_cue",
        "S 56": "stimulus_A_postreversalPunishment",
        "S 60": "feedback_positive",
        "S 61": "feedback_negative",
    },
    continuous_cols=["response_time"],
)
