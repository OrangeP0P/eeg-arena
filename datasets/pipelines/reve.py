"""REVE-style Raw-level preprocessing preset.

Implements resampling to 200 Hz, 0.5-99.5 Hz band-pass with Nyquist safety,
float32 data storage, per-channel recording z-score, and clipping to +/-15.

Omitted by design: filtering out recordings shorter than 10 s, excluding
downstream test samples from pretraining data, and 1 s/0.1 s patch windows.
Those are dataset selection and model input concerns.
"""
from datasets.pipelines._fm_common import (
    cast_raw_float32,
    clip_picks,
    data_channel_indices,
    per_channel_zscore,
    safe_filter,
    safe_resample,
)


def make_pipeline(
    resample: float = 200.0,
    lower_bound: float = 0.5,
    upper_bound: float = 99.5,
    normalize: bool = True,
    clip: float = 15.0,
):
    """Return a ``pipeline(raw) -> raw`` closure for REVE."""

    def pipeline(raw):
        safe_resample(raw, resample)
        picks = data_channel_indices(raw)
        safe_filter(raw, picks, lower_bound=lower_bound, upper_bound=upper_bound)
        cast_raw_float32(raw)
        picks = data_channel_indices(raw)
        if normalize:
            per_channel_zscore(raw, picks)
        if clip is not None and float(clip) > 0:
            clip_picks(raw, picks, -float(clip), float(clip))
        return raw

    return pipeline
