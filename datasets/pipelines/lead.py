"""LEAD-style Raw-level preprocessing preset.

Implements 0.5-45 Hz band-pass, resampling to 128 Hz, and per-channel z-score
normalization.

Omitted by design: ICA/ICLabel artifact removal, spherical-spline/coordinate
channel interpolation, fixed 19-channel montage alignment, and multi-scale
window construction. Those belong to data-cleaning, data/channels.py, or
dataset epoch configuration.
"""
from datasets.pipelines._fm_common import data_channel_indices, per_channel_zscore, safe_filter, safe_resample


def make_pipeline(
    resample: float = 128.0,
    lower_bound: float = 0.5,
    upper_bound: float = 45.0,
    normalize: bool = True,
):
    """Return a ``pipeline(raw) -> raw`` closure for LEAD."""

    def pipeline(raw):
        picks = data_channel_indices(raw)
        safe_filter(raw, picks, lower_bound=lower_bound, upper_bound=upper_bound)
        safe_resample(raw, resample)
        picks = data_channel_indices(raw)
        if normalize:
            per_channel_zscore(raw, picks)
        return raw

    return pipeline
