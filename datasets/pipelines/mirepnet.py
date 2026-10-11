"""MIRepNet-style Raw-level preprocessing preset.

Implements the motor-imagery band-pass of 8-30 Hz and resampling to 250 Hz.

Omitted by design: channel-template interpolation and Euclidean Alignment.
Those are data-layer/channel-space and trial-level alignment steps, not Raw
pipeline operations.
"""
from datasets.pipelines._fm_common import data_channel_indices, safe_filter, safe_resample


def make_pipeline(
    resample: float = 250.0,
    lower_bound: float = 8.0,
    upper_bound: float = 30.0,
):
    """Return a ``pipeline(raw) -> raw`` closure for MIRepNet."""

    def pipeline(raw):
        picks = data_channel_indices(raw)
        safe_filter(raw, picks, lower_bound=lower_bound, upper_bound=upper_bound)
        safe_resample(raw, resample)
        return raw

    return pipeline
