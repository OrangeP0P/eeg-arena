"""BENDR-style Raw-level preprocessing preset.

Implements optional low-pass filtering for P300-style high-sampling-rate data,
resampling to 256 Hz, and per-recording min-max scaling to [-1, 1].

Omitted by design: task/window cropping, Deep10-10/UI 10-20 channel-space
selection, zero-filling missing channels, and BENDR's extra relative-amplitude
channel. Those belong to dataset epoch config, data/channels.py, or the model
adapter rather than this Raw pipeline.
"""
from datasets.pipelines._fm_common import (
    data_channel_indices,
    per_recording_minmax,
    safe_filter,
    safe_resample,
)


def make_pipeline(resample: float = 256.0, upper_bound=None):
    """Return a ``pipeline(raw) -> raw`` closure for BENDR."""

    def pipeline(raw):
        picks = data_channel_indices(raw)
        safe_filter(raw, picks, upper_bound=upper_bound)
        safe_resample(raw, resample)
        picks = data_channel_indices(raw)
        per_recording_minmax(raw, picks)
        return raw

    return pipeline
