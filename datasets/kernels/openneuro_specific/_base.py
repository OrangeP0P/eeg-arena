"""Base kernel class for OpenNeuro dataset-specific kernels.

Extends BIDSKernel with:
- Automatic response_time misc channel injection from events.tsv
- Event annotation recoding utilities
- STIM channel construction from continuous variables

All dataset-specific kernels in this package subclass OpenNeuroBaseKernel.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import mne
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from kernels.bids import BIDSKernel, _find_events_tsv


@dataclass
class OpenNeuroBaseKernel(BIDSKernel):
    """Extended BIDS kernel for OpenNeuro datasets.

    In addition to the standard BIDS metadata extraction, this kernel:

    1. Reads continuous variables from events.tsv (e.g. response_time,
       confidence, gabor_contrast) and adds them as ``misc`` channels
       aligned to the EEG timeline.

    2. Provides ``recode_annotations()`` for subclasses to remap numerical
       or cryptic event codes to meaningful string labels.

    3. Provides ``add_stim_channel()`` for creating a new STIM-type channel
       from a per-sample numeric array.

    Subclasses override ``event_map`` (dict) and/or ``continuous_cols``
    (list of events.tsv column names to inject as misc channels).
    """

    KERNEL_ID: str = "openneuro-base"

    #: Mapping from raw annotation labels to human-readable labels.
    #: Override in subclasses. e.g. {"1": "left_hand", "2": "right_hand"}
    event_map: Dict[str, str] = field(default_factory=dict)

    #: events.tsv columns to inject as misc channels.
    #: Override in subclasses. Always tries "response_time" by default.
    continuous_cols: List[str] = field(default_factory=lambda: ["response_time"])

    # ------------------------------------------------------------------
    # Misc / STIM channel utilities
    # ------------------------------------------------------------------

    def _load_events_df(self, file_path: str) -> Optional[pd.DataFrame]:
        """Load the BIDS events.tsv associated with file_path, or None."""
        events_path = _find_events_tsv(file_path)
        if events_path is None:
            return None
        try:
            return pd.read_csv(events_path, sep="\t")
        except Exception:
            return None

    def inject_misc_channels(
        self,
        raw: mne.io.BaseRaw,
        file_path: str,
        cols: Optional[List[str]] = None,
    ) -> mne.io.BaseRaw:
        """Add continuous variables from events.tsv as misc channels.

        For each column in *cols* that exists in the events.tsv and has at
        least one non-NaN value, a new ``misc`` channel is created and filled
        with per-sample values via nearest-onset interpolation (step function:
        the value is held constant from one event onset to the next).

        Parameters
        ----------
        raw : mne.io.BaseRaw
            The raw object to modify (must be preloaded).
        file_path : str
            Path to the EEG file (used to locate events.tsv).
        cols : list of str, optional
            Column names to inject.  Defaults to ``self.continuous_cols``.

        Returns
        -------
        mne.io.BaseRaw
            Raw object with new misc channels appended.
        """
        if cols is None:
            cols = self.continuous_cols

        ev_df = self._load_events_df(file_path)
        if ev_df is None or ev_df.empty:
            return raw

        sfreq = raw.info["sfreq"]
        n_times = raw.n_times

        for col in cols:
            if col not in ev_df.columns:
                continue
            col_data = ev_df[col].copy()
            if col_data.dropna().empty:
                continue

            # Build a per-sample array (step function from onset)
            ch_data = np.full(n_times, np.nan)
            for _, event_row in ev_df.iterrows():
                onset_sec = float(event_row.get("onset", np.nan))
                if np.isnan(onset_sec):
                    continue
                val = event_row[col]
                if pd.isna(val):
                    continue
                sample_idx = int(round(onset_sec * sfreq))
                if 0 <= sample_idx < n_times:
                    ch_data[sample_idx] = float(val)

            # Forward-fill NaN → step function
            last_valid = np.nan
            for i in range(n_times):
                if not np.isnan(ch_data[i]):
                    last_valid = ch_data[i]
                elif not np.isnan(last_valid):
                    ch_data[i] = last_valid

            # Create info for new misc channel
            ch_name = col if col not in raw.ch_names else f"misc_{col}"
            misc_info = mne.create_info(
                ch_names=[ch_name],
                sfreq=sfreq,
                ch_types=["misc"],
            )
            misc_raw = mne.io.RawArray(ch_data[np.newaxis, :], misc_info, verbose=False)

            # Set measurement date to match
            misc_raw.set_meas_date(raw.info.get("meas_date"))
            raw.load_data(verbose=False)
            raw.add_channels([misc_raw], force_update_info=True)

        return raw

    def add_stim_channel(
        self,
        raw: mne.io.BaseRaw,
        ch_name: str,
        data: np.ndarray,
    ) -> mne.io.BaseRaw:
        """Append a STIM channel to raw.

        Parameters
        ----------
        raw : mne.io.BaseRaw
        ch_name : str
            Name for the new channel.
        data : np.ndarray, shape (n_times,)
            Integer trigger values per sample.

        Returns
        -------
        mne.io.BaseRaw
        """
        stim_info = mne.create_info(
            ch_names=[ch_name],
            sfreq=raw.info["sfreq"],
            ch_types=["stim"],
        )
        stim_raw = mne.io.RawArray(data[np.newaxis, :].astype(float), stim_info, verbose=False)
        stim_raw.set_meas_date(raw.info.get("meas_date"))
        raw.load_data(verbose=False)
        raw.add_channels([stim_raw], force_update_info=True)
        return raw

    # ------------------------------------------------------------------
    # Event recoding
    # ------------------------------------------------------------------

    def recode_annotations(
        self,
        raw: mne.io.BaseRaw,
        event_map: Optional[Dict[str, str]] = None,
    ) -> mne.io.BaseRaw:
        """Remap annotation labels using event_map.

        Strips trailing '.0' from float-encoded labels before lookup.
        Unknown labels are kept unchanged.

        Parameters
        ----------
        raw : mne.io.BaseRaw
        event_map : dict, optional
            Override ``self.event_map`` for this call.

        Returns
        -------
        mne.io.BaseRaw
        """
        mapping = event_map if event_map is not None else self.event_map
        if not mapping:
            return raw

        new_descriptions = []
        for desc in raw.annotations.description:
            key = str(desc).strip()
            # Normalise float codes like "1.0" → "1"
            key_norm = re.sub(r'\.0$', '', key)
            label = mapping.get(key, mapping.get(key_norm, desc))
            new_descriptions.append(label)

        raw.annotations.description[:] = new_descriptions
        return raw

    # ------------------------------------------------------------------
    # EEGUnity kernel interface
    # ------------------------------------------------------------------

    def apply(self, udataset, raw: mne.io.BaseRaw, row) -> mne.io.BaseRaw:
        """Apply BIDS metadata + event recoding + misc channel injection."""
        # 1. Standard BIDS metadata (age, sex, events summary)
        raw = super().apply(udataset, raw, row)

        # 2. Recode annotations if subclass provided an event_map
        if self.event_map:
            raw = self.recode_annotations(raw)

        # 3. Inject misc channels from events.tsv continuous columns
        if self.continuous_cols:
            try:
                raw = self.inject_misc_channels(raw, row["File Path"])
            except Exception:
                pass  # Fail silently; misc channels are enrichment only

        return raw
