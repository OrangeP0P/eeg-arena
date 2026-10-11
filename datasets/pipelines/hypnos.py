"""Hypnos' causal EEG preprocessing preset.

Matches the released tokenizer path: 128 Hz, forward-only 0.5--45 Hz IIR,
50/60 Hz notch, causal rolling-EMA z-score (60 s), then logarithmic amplitude
compression beyond eight standard deviations.
"""

import numpy as np
from scipy.signal import butter, iirnotch, lfilter, lfilter_zi, sosfilt, sosfilt_zi, tf2sos

from datasets.pipelines._fm_common import (
    cast_raw_float32,
    data_channel_indices,
    safe_resample,
    scale_picks,
)


def _causal_sos(sos, signal):
    zi = sosfilt_zi(sos) * float(signal[0])
    output, _ = sosfilt(sos, signal, zi=zi)
    return output


def _rolling_normalize(
    signal, fs, tau_seconds=60.0, clip_sigma=6.0, eps=1e-6
):
    values = np.asarray(signal, dtype=np.float64)
    alpha = float(np.exp(-1.0 / max(tau_seconds * fs, 1.0)))
    b = np.array([1.0 - alpha], dtype=np.float64)
    a = np.array([1.0, -alpha], dtype=np.float64)
    zi = lfilter_zi(b, a)
    initial = values[: min(len(values), max(1, int(tau_seconds * fs)))]
    mean0 = float(np.nanmean(initial))
    var0 = max(float(np.nanvar(initial)), eps)
    safe = np.where(np.isnan(values), mean0, values)
    mean, _ = lfilter(b, a, safe, zi=zi * mean0)
    square = (safe - mean) ** 2
    variance, _ = lfilter(b, a, square, zi=zi * var0)
    variance = np.maximum(variance, eps)
    clipped_square = np.minimum(square, float(clip_sigma) ** 2 * variance)
    variance, _ = lfilter(b, a, clipped_square, zi=zi * var0)
    return (values - mean) / np.sqrt(np.maximum(variance, eps))


def _compress(signal, knee=8.0, scale=2.0):
    output = signal.copy()
    positive = signal > knee
    negative = signal < -knee
    output[positive] = knee + scale * np.log1p((signal[positive] - knee) / scale)
    output[negative] = -knee - scale * np.log1p((-signal[negative] - knee) / scale)
    return output


def make_pipeline(resample=128.0, notch=50.0, tau_seconds=60.0, scale_uv=True):
    def pipeline(raw):
        safe_resample(raw, resample)
        raw.load_data()
        picks = data_channel_indices(raw)
        # Hypnos reads EDF physical values (microvolts for EEG), whereas MNE
        # stores EEG in volts.  Convert before the variance-floored rolling
        # normalizer so ordinary EEG is not collapsed by its 1e-6 epsilon.
        if scale_uv:
            scale_picks(raw, picks, 1e6)
        fs = float(raw.info["sfreq"])
        band = butter(4, [0.5, 45.0], btype="bandpass", fs=fs, output="sos")
        notch_sos = None
        if notch and 0.0 < float(notch) < fs / 2.0:
            b, a = iirnotch(float(notch), 30.0, fs)
            notch_sos = tf2sos(b, a)
        for pick in picks:
            signal = _causal_sos(band, raw._data[pick].astype(np.float64))
            if notch_sos is not None:
                signal = _causal_sos(notch_sos, signal)
            signal = _rolling_normalize(signal, fs, tau_seconds=tau_seconds)
            raw._data[pick] = _compress(signal).astype(np.float32)
        return cast_raw_float32(raw)

    return pipeline
