"""Dataset-specific kernel for openneuro ds003190.

Assessment of visual stimuli properties in P300 ERP.
Events.tsv has value column (1-8 codes) with type=trigger.
P300 speller matrix paradigm: 6 rows + 6 columns of a 6x6 matrix = 12 classes,
but simplified to 8 event types here.
  1 = row/column flash containing target
  2 = row/column flash not containing target
  3-8 = specific row/column identifiers
TODO: verify exact mapping from paradigm description.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from kernels.openneuro_specific._base import OpenNeuroBaseKernel

KERNEL = OpenNeuroBaseKernel(
    KERNEL_ID="openneuro-ds003190",
    event_map={
"1": "target_flash",
        "2": "nontarget_flash",
        "3": "flash_type_3",
        "4": "flash_type_4",
        "5": "flash_type_5",
        "6": "flash_type_6",
        "7": "flash_type_7",
        "8": "flash_type_8",
    },
    continuous_cols=["response_time"],
)
