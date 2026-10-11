"""
Kernel for OpenBMI dataset (other_openbmi).

Dataset: https://gigadb.org/dataset/view/id/100542
Format: MATLAB .mat files, one per session per subject per task
  Naming: sess{01|02}_subj{01..54}_EEG_{MI|SSVEP|ERP|Artifact}.mat

This kernel handles EEG_MI files (Motor Imagery — left vs right hand).
Each .mat contains EEG_MI_train and EEG_MI_test structs with fields:
  - x: (n_samples, 62) continuous EEG
  - t: (n_trials,) sample indices of trial onset
  - y_class: (n_trials,) string labels ('left' / 'right')
  - fs: sampling frequency (1000 Hz)
  - chan: (62,) channel names
  - time_interval: [0, 4000] ms — full epoch window

The kernel only applies to EEG_MI files. Non-MI files (SSVEP, ERP, Artifact)
are loaded by the default EEGUnity parser and receive no annotation injection.

Annotation labels produced:
  - 'left hand MI'  — left-hand motor imagery trial onset
  - 'right hand MI' — right-hand motor imagery trial onset

Since each .mat packs train+test blocks sequentially, this kernel:
  1. Reads train block (x_train) and appends test block (x_test).
  2. Creates one continuous RawArray.
  3. Sets annotations for all trials from both blocks.

Subject demographics (Lee et al., 2019 — Table 1):
  Ages 19–32, mixed sex. Full table encoded below.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import scipy.io as scio


# ---------------------------------------------------------------------------
# Subject demographics (54 subjects, from Lee et al. 2019)
# ---------------------------------------------------------------------------
_SUBJECT_INFO = {
    "subject01": {"age": "24", "sex": "Female"},
    "subject02": {"age": "24", "sex": "Female"},
    "subject03": {"age": "22", "sex": "Female"},
    "subject04": {"age": "21", "sex": "Female"},
    "subject05": {"age": "25", "sex": "Male"},
    "subject06": {"age": "26", "sex": "Female"},
    "subject07": {"age": "24", "sex": "Male"},
    "subject08": {"age": "22", "sex": "Female"},
    "subject09": {"age": "25", "sex": "Male"},
    "subject10": {"age": "24", "sex": "Female"},
    "subject11": {"age": "21", "sex": "Female"},
    "subject12": {"age": "23", "sex": "Female"},
    "subject13": {"age": "22", "sex": "Male"},
    "subject14": {"age": "25", "sex": "Male"},
    "subject15": {"age": "32", "sex": "Male"},
    "subject16": {"age": "24", "sex": "Male"},
    "subject17": {"age": "22", "sex": "Female"},
    "subject18": {"age": "19", "sex": "Female"},
    "subject19": {"age": "20", "sex": "Female"},
    "subject20": {"age": "25", "sex": "Male"},
    "subject21": {"age": "23", "sex": "Female"},
    "subject22": {"age": "25", "sex": "Female"},
    "subject23": {"age": "26", "sex": "Male"},
    "subject24": {"age": "27", "sex": "Male"},
    "subject25": {"age": "25", "sex": "Male"},
    "subject26": {"age": "27", "sex": "Male"},
    "subject27": {"age": "20", "sex": "Male"},
    "subject28": {"age": "21", "sex": "Male"},
    "subject29": {"age": "23", "sex": "Male"},
    "subject30": {"age": "28", "sex": "Female"},
    "subject31": {"age": "23", "sex": "Female"},
    "subject32": {"age": "29", "sex": "Male"},
    "subject33": {"age": "23", "sex": "Male"},
    "subject34": {"age": "25", "sex": "Male"},
    "subject35": {"age": "23", "sex": "Male"},
    "subject36": {"age": "22", "sex": "Female"},
    "subject37": {"age": "26", "sex": "Male"},
    "subject38": {"age": "19", "sex": "Male"},
    "subject39": {"age": "25", "sex": "Female"},
    "subject40": {"age": "23", "sex": "Male"},
    "subject41": {"age": "31", "sex": "Male"},
    "subject42": {"age": "22", "sex": "Male"},
    "subject43": {"age": "26", "sex": "Female"},
    "subject44": {"age": "26", "sex": "Female"},
    "subject45": {"age": "22", "sex": "Male"},
    "subject46": {"age": "21", "sex": "Female"},
    "subject47": {"age": "29", "sex": "Female"},
    "subject48": {"age": "27", "sex": "Male"},
    "subject49": {"age": "25", "sex": "Female"},
    "subject50": {"age": "19", "sex": "Male"},
    "subject51": {"age": "28", "sex": "Male"},
    "subject52": {"age": "32", "sex": "Male"},
    "subject53": {"age": "25", "sex": "Male"},
    "subject54": {"age": "23", "sex": "Male"},
}

_LABEL_MAP = {
    "left": "left hand MI",
    "right": "right hand MI",
}

# MI epoch window is [0, 4000] ms per dataset spec
_EPOCH_DURATION_S = 4.0


def _derive_subject_key(file_path: str) -> str:
    """Extract subject key from filename like 'sess01_subj03_EEG_MI.mat'."""
    fname = Path(file_path).stem  # e.g. sess01_subj03_EEG_MI
    parts = fname.split("_")
    # parts[1] == 'subj03'
    subj_raw = parts[1]  # subj03
    num = subj_raw[4:]   # '03'
    return f"subject{num.zfill(2)}"


def _is_mi_file(file_path: str) -> bool:
    return "_EEG_MI" in Path(file_path).name


def _normalize_channel_names(raw: mne.io.BaseRaw, block: dict) -> list[str]:
    channel_values = np.asarray(block.get("chan", [])).flatten()
    channel_names = [str(ch).strip() for ch in channel_values if str(ch).strip()]
    if channel_names:
        return channel_names
    return list(raw.ch_names)


@dataclass
class OpenBMIKernel:
    """Kernel for OpenBMI MI dataset.

    Injects:
    - Subject metadata (age, sex, amplifier) into raw.info['description']
    - Annotations for all MI trials (train + test blocks concatenated)
    """

    KERNEL_ID: str = "other-openbmi-mi-v1"

    def apply(self, udataset, raw: mne.io.BaseRaw, row) -> mne.io.BaseRaw:
        file_path = row["File Path"]

        # Only enrich MI files; pass through others unchanged
        if not _is_mi_file(file_path):
            return raw

        # --- Subject metadata ---
        subj_key = _derive_subject_key(file_path)
        subj_info = _SUBJECT_INFO.get(subj_key, {})
        description_dict = {
            "original_description": raw.info.get("description", ""),
            "eegunity_description": {
                "amplifier": "BrainAmp (Brain Products; Munich, Germany)",
                "cap": "Ag/AgCl, 62-channel, 10-20 system",
                "age": subj_info.get("age", "unknown"),
                "sex": subj_info.get("sex", "unknown"),
                "handedness": "unknown",
            },
        }
        raw.info["description"] = json.dumps(description_dict)

        # --- Load .mat and inject annotations ---
        try:
            mat = scio.loadmat(file_path, simplify_cells=True)
        except Exception:
            # If loadmat fails (e.g. already-converted fif), return as-is
            return raw

        train_block = mat.get("EEG_MI_train")
        if train_block is None:
            return raw

        sfreq = float(train_block.get("fs", raw.info["sfreq"]))
        channel_names = _normalize_channel_names(raw, train_block)
        n_channels = len(channel_names)
        blocks = []
        onsets, durations, descriptions = [], [], []
        sample_offset = 0

        for block_name in ("EEG_MI_train", "EEG_MI_test"):
            block = mat.get(block_name)
            if block is None:
                continue

            block_data = np.asarray(block["x"], dtype=float)
            if block_data.ndim != 2:
                raise ValueError(f"Unexpected OpenBMI block shape in {block_name}: {block_data.shape}")
            if block_data.shape[1] != n_channels and block_data.shape[0] == n_channels:
                block_data = block_data.T
            if block_data.shape[1] != n_channels:
                raise ValueError(
                    f"OpenBMI channel count mismatch in {block_name}: "
                    f"data has {block_data.shape[1]}, metadata has {n_channels}"
                )

            t = np.asarray(block["t"]).flatten()
            y_class = np.asarray(block["y_class"]).flatten()

            for sample_idx, label in zip(t, y_class):
                desc = _LABEL_MAP.get(str(label).strip().lower(), str(label))
                if desc not in _LABEL_MAP.values():
                    continue
                onset_s = (sample_offset + float(sample_idx)) / sfreq
                onsets.append(onset_s)
                durations.append(_EPOCH_DURATION_S)
                descriptions.append(desc)

            blocks.append(block_data)
            sample_offset += block_data.shape[0]

        if not blocks:
            return raw

        data = np.concatenate(blocks, axis=0).T
        info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info, verbose="ERROR")
        raw.info["description"] = json.dumps(description_dict)

        if onsets:
            ann = mne.Annotations(
                onset=onsets,
                duration=durations,
                description=descriptions,
            )
            raw.set_annotations(ann)

        return raw


KERNEL = OpenBMIKernel()
