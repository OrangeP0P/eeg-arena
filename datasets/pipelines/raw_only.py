"""
Raw-only pipeline (no filtering, no normalization).

Only resampling (handled by the epoch functions themselves) is applied.
Useful for debugging or when the raw data is already preprocessed.
"""


def pipeline(raw):
    """
    Pass-through pipeline: returns raw unchanged.

    Args:
        raw: mne.io.Raw object

    Returns:
        Unchanged mne.io.Raw object
    """
    return raw
