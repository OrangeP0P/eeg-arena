"""Shared helpers for EEG foundation-model preprocessing presets.

The public CLI entry points live in per-model files in this directory.  This
module keeps the low-level Raw operations consistent across those presets.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Optional, Sequence

import numpy as np


def data_channel_indices(raw):
    """Return channels to transform while preserving EEGUnity labels/triggers."""
    try:
        from eegunity.utils.label_channel import misc_channel_indices, stim_channel_indices

        skip_idx = set(misc_channel_indices(raw)) | set(stim_channel_indices(raw))
    except Exception:
        skip_idx = set()
    return [idx for idx in range(len(raw.ch_names)) if idx not in skip_idx]


def safe_resample(raw, resample: Optional[float]):
    """Resample while keeping misc label channels nearest-neighbor aligned."""
    if resample is None:
        return raw
    target = float(resample)
    if target <= 0:
        return raw

    current = float(raw.info["sfreq"])
    if abs(current - target) <= 1e-6:
        return raw

    from eegunity.utils.label_channel import resample_raw_with_labels

    return resample_raw_with_labels(raw, sfreq=target, npad="auto", verbose=False)


def _positive_float(value):
    if value is None:
        return None
    value = float(value)
    return value if value > 0 else None


def safe_filter(
    raw,
    picks,
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
    *,
    method: str = "fir",
):
    """Apply a Nyquist-safe filter to selected data channels."""
    if not picks:
        return raw

    sfreq = float(raw.info["sfreq"])
    nyquist = sfreq / 2.0
    eps = max(1e-6, nyquist * 1e-6)

    l_freq = _positive_float(lower_bound)
    h_freq = _positive_float(upper_bound)

    if l_freq is not None and l_freq >= nyquist - eps:
        l_freq = None
    if h_freq is not None:
        h_freq = min(h_freq, nyquist - eps)
        if h_freq <= 0:
            h_freq = None
    if l_freq is not None and h_freq is not None and h_freq <= l_freq:
        h_freq = None
    if l_freq is None and h_freq is None:
        return raw

    raw.filter(l_freq=l_freq, h_freq=h_freq, picks=picks, method=method, verbose=False)
    return raw


def _iter_notch_freqs(notch) -> Sequence[float]:
    if notch is None:
        return ()
    if isinstance(notch, str):
        cleaned = notch.strip()
        if not cleaned or cleaned.lower() in {"none", "false", "0"}:
            return ()
        return tuple(float(part.strip()) for part in cleaned.split(",") if part.strip())
    if isinstance(notch, Iterable):
        return tuple(float(freq) for freq in notch)
    return (float(notch),)


def safe_notch(raw, picks, notch):
    """Apply a Nyquist-safe notch filter to selected data channels."""
    if not picks:
        return raw

    sfreq = float(raw.info["sfreq"])
    nyquist = sfreq / 2.0
    freqs = [freq for freq in _iter_notch_freqs(notch) if 0.0 < freq < nyquist]
    if not freqs:
        return raw

    raw.notch_filter(freqs=freqs, picks=picks, verbose=False)
    return raw


def set_average_reference(raw):
    """Average-reference EEG channels when MNE channel metadata supports it."""
    try:
        raw.set_eeg_reference("average", projection=False, verbose=False)
    except Exception:
        pass
    return raw


def scale_picks(raw, picks, factor: float):
    if not picks:
        return raw
    raw.load_data()
    raw._data[picks] *= float(factor)
    return raw


def per_recording_minmax(raw, picks, eps: float = 1e-8):
    """Scale all selected channels in one Raw object to approximately [-1, 1]."""
    if not picks:
        return raw
    raw.load_data()
    data = raw._data[picks]
    d_min = float(np.nanmin(data))
    d_max = float(np.nanmax(data))
    span = d_max - d_min
    if not np.isfinite(span) or span <= eps:
        raw._data[picks] = 0.0
    else:
        raw._data[picks] = 2.0 * ((data - d_min) / (span + eps)) - 1.0
    return raw


def per_channel_abs_quantile(raw, picks, quantile: float = 0.95, eps: float = 1e-8):
    if not picks:
        return raw
    raw.load_data()
    data = raw._data[picks]
    denom = np.nanquantile(np.abs(data), float(quantile), axis=1, keepdims=True)
    denom = np.where(np.isfinite(denom) & (denom > eps), denom, 1.0)
    raw._data[picks] = data / denom
    return raw


def per_channel_zscore(raw, picks, eps: float = 1e-8):
    if not picks:
        return raw
    raw.load_data()
    data = raw._data[picks]
    mean = np.nanmean(data, axis=1, keepdims=True)
    std = np.nanstd(data, axis=1, keepdims=True)
    std = np.where(np.isfinite(std) & (std > eps), std, 1.0)
    raw._data[picks] = (data - mean) / std
    return raw


def remove_dc_offset(raw, picks):
    if not picks:
        return raw
    raw.load_data()
    raw._data[picks] -= np.nanmean(raw._data[picks], axis=1, keepdims=True)
    return raw


def linear_detrend(raw, picks):
    if not picks:
        return raw
    raw.load_data()
    from scipy import signal

    raw._data[picks] = signal.detrend(raw._data[picks], axis=1, type="linear")
    return raw


def clip_picks(raw, picks, lower: float, upper: float):
    if not picks:
        return raw
    raw.load_data()
    raw._data[picks] = np.clip(raw._data[picks], float(lower), float(upper))
    return raw


def cast_raw_float32(raw):
    raw.load_data()
    raw._data = raw._data.astype(np.float32, copy=False)
    return raw
