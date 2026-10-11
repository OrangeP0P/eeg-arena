"""
Classical EEG preprocessing pipeline with z-score normalization.

Exports ``make_pipeline(resample, lower_bound, upper_bound, notch)``
which returns a ``pipeline(raw) -> raw`` closure suitable for EEGUnity.

Usage in CLI
------------
    --pipeline pipelines/classical_pipeline_with_z_score.py
    --pipeline_param resample=200 lower_bound=0.1 upper_bound=75 notch=50

Steps applied (in order)
------------------------
1. Resample to ``resample`` Hz
2. Bandpass filter: [``lower_bound``, ``upper_bound``] Hz
3. Notch filter at ``notch`` Hz
4. Per-channel z-score normalization
"""
import sys
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def make_pipeline(resample: float = 200, lower_bound: float = 0.1,
                  upper_bound: float = 75.0, notch: float = 50.0):
    """
    Factory that returns a preprocessing pipeline closure.

    Parameters
    ----------
    resample : float
        Target sampling frequency in Hz.
    lower_bound : float
        High-pass cutoff frequency in Hz.
    upper_bound : float
        Low-pass cutoff frequency in Hz.
    notch : float
        Notch filter frequency in Hz.

    Returns
    -------
    Callable[[mne.io.Raw], mne.io.Raw]
    """
    def pipeline(raw):
        from eegunity.utils.normalize import normalize_mne

        raw.resample(sfreq=resample, verbose=False)
        raw.filter(l_freq=lower_bound, h_freq=upper_bound, verbose=False)
        raw.notch_filter(freqs=notch, verbose=False)
        return normalize_mne(raw)

    return pipeline
