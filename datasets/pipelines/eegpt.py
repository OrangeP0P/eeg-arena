"""EEGPT-style Raw-level preprocessing preset.

Implements average reference, optional task-specific band/low-pass filtering,
resampling to 256 Hz, and conversion from volts to millivolts.

Omitted by design: task window selection, Euclidean Alignment, channel
dictionary enforcement, and non-overlapping 250 ms patch construction. Those
belong to dataset/task config, data/channels.py, or model input handling.
"""
from datasets.pipelines._fm_common import (
    data_channel_indices,
    safe_filter,
    safe_resample,
    scale_picks,
    set_average_reference,
)


def make_pipeline(
    resample: float = 256.0,
    lower_bound=None,
    upper_bound=None,
    scale_mv: bool = True,
):
    """Return a ``pipeline(raw) -> raw`` closure for EEGPT."""

    def pipeline(raw):
        set_average_reference(raw)
        safe_resample(raw, resample)
        picks = data_channel_indices(raw)
        safe_filter(raw, picks, lower_bound=lower_bound, upper_bound=upper_bound)
        if scale_mv:
            scale_picks(raw, picks, 1e3)
        return raw

    return pipeline
