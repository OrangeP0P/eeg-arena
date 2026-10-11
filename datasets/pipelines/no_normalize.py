"""
Filter-only pipeline (no normalization).

Applies:
1. Adaptive bandpass filter (l_freq=0.1, h_freq adaptive)
2. Adaptive notch filter

Skips per-channel normalization. Useful when normalization is done
in the model or during training.
"""


def pipeline(raw):
    """
    Apply bandpass filter + notch filter only (no normalization).

    Args:
        raw: mne.io.Raw object

    Returns:
        Filtered mne.io.Raw object
    """
    sfreq = raw.info['sfreq']
    h_freq = (sfreq / 2) - 1 if sfreq <= 150 else 75
    notch_freq = (sfreq / 2) - 1 if sfreq <= 100 else 50

    raw.filter(l_freq=0.1, h_freq=h_freq, verbose=False)
    raw.notch_filter(freqs=notch_freq, verbose=False)
    return raw
