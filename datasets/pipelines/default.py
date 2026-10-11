"""
Default preprocessing pipeline.

Applies:
1. Adaptive bandpass filter (l_freq=0.1, h_freq adaptive based on sfreq)
2. Adaptive notch filter (50 or adaptive based on sfreq)
3. Per-channel zero-mean / unit-variance normalization

This is equivalent to ELIOT stage5's build_pipeline() default behavior.
"""
import sys
import os

# Ensure eegunity from the EEG-Arena project root is used
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def pipeline(raw):
    """
    Apply bandpass filter + notch filter + per-channel normalization.

    Args:
        raw: mne.io.Raw object

    Returns:
        Preprocessed mne.io.Raw object
    """
    from eegunity.utils.normalize import normalize_mne

    sfreq = raw.info['sfreq']
    h_freq = (sfreq / 2) - 1 if sfreq <= 150 else 75
    notch_freq = (sfreq / 2) - 1 if sfreq <= 100 else 50

    raw.filter(l_freq=0.1, h_freq=h_freq, verbose=False)
    raw.notch_filter(freqs=notch_freq, verbose=False)
    return normalize_mne(raw)
