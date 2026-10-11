"""ST-EEGFormer-style Raw-level preprocessing preset.

Implements configurable power-line notch, 0.1-64 Hz FIR band-pass, resampling
to 128 Hz, and per-channel z-score normalization.

Omitted by design: continuous/SSVEP-specific windowing and fixed channel
vocabulary handling. Those belong to dataset epoch settings or data/channels.py.
"""
from datasets.pipelines._fm_common import data_channel_indices, per_channel_zscore, safe_filter, safe_notch, safe_resample


def make_pipeline(
    resample: float = 128.0,
    lower_bound: float = 0.1,
    upper_bound: float = 64.0,
    notch: float = 50.0,
    normalize: bool = True,
):
    """Return a ``pipeline(raw) -> raw`` closure for ST-EEGFormer."""

    def pipeline(raw):
        picks = data_channel_indices(raw)
        safe_notch(raw, picks, notch)
        safe_filter(raw, picks, lower_bound=lower_bound, upper_bound=upper_bound, method="fir")
        safe_resample(raw, resample)
        picks = data_channel_indices(raw)
        if normalize:
            per_channel_zscore(raw, picks)
        return raw

    return pipeline
