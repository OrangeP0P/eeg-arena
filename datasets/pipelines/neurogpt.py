"""NeuroGPT-style Raw-level preprocessing preset.

Implements average reference, 60 Hz notch, 0.5-100 Hz band-pass, resampling to
250 Hz, DC offset correction, linear detrending, and per-channel z-transform.

Omitted by design: bad-channel interpolation, fixed 22-channel layout
projection, 2 s chunking, 10% overlap, and downstream MI transformation
matrices. Those are channel-space or dataset epoch responsibilities.
"""
from datasets.pipelines._fm_common import (
    data_channel_indices,
    linear_detrend,
    per_channel_zscore,
    remove_dc_offset,
    safe_filter,
    safe_notch,
    safe_resample,
    set_average_reference,
)


def make_pipeline(
    resample: float = 250.0,
    lower_bound: float = 0.5,
    upper_bound: float = 100.0,
    notch: float = 60.0,
    normalize: bool = True,
):
    """Return a ``pipeline(raw) -> raw`` closure for NeuroGPT."""

    def pipeline(raw):
        set_average_reference(raw)
        picks = data_channel_indices(raw)
        safe_notch(raw, picks, notch)
        safe_filter(raw, picks, lower_bound=lower_bound, upper_bound=upper_bound)
        safe_resample(raw, resample)
        picks = data_channel_indices(raw)
        remove_dc_offset(raw, picks)
        linear_detrend(raw, picks)
        if normalize:
            per_channel_zscore(raw, picks)
        return raw

    return pipeline
