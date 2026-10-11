"""LUNA-style Raw-level preprocessing preset.

Implements 0.1-75 Hz band-pass, configurable 50/60 Hz notch, resampling to
256 Hz, and per-channel z-score normalization.

Omitted by design: bipolar montage construction for specific TUH datasets and
temporal patch construction. Those are channel-space/model input concerns.
"""
from datasets.pipelines._fm_common import data_channel_indices, per_channel_zscore, safe_filter, safe_notch, safe_resample


def make_pipeline(
    resample: float = 256.0,
    lower_bound: float = 0.1,
    upper_bound: float = 75.0,
    notch: float = 50.0,
    normalize: bool = True,
):
    """Return a ``pipeline(raw) -> raw`` closure for LUNA."""

    def pipeline(raw):
        picks = data_channel_indices(raw)
        safe_filter(raw, picks, lower_bound=lower_bound, upper_bound=upper_bound)
        safe_notch(raw, picks, notch)
        safe_resample(raw, resample)
        picks = data_channel_indices(raw)
        if normalize:
            per_channel_zscore(raw, picks)
        return raw

    return pipeline
