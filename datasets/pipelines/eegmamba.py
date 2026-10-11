"""EEGMamba-style Raw-level preprocessing preset.

Implements 0.3-75 Hz band-pass, configurable power-line notch, resampling to
200 Hz, and scaling from volts to units of 100 microvolts.

Omitted by design: 30 s segmentation, 1 s non-overlapping patching, and any
channel-space preset. Those are handled by dataset epochs, model input
formatting, or data/channels.py.
"""
from datasets.pipelines._fm_common import data_channel_indices, safe_filter, safe_notch, safe_resample, scale_picks


def make_pipeline(
    resample: float = 200.0,
    lower_bound: float = 0.3,
    upper_bound: float = 75.0,
    notch: float = 50.0,
    scale_uv: bool = True,
    uv_scale: float = 100.0,
):
    """Return a ``pipeline(raw) -> raw`` closure for EEGMamba."""

    def pipeline(raw):
        picks = data_channel_indices(raw)
        safe_filter(raw, picks, lower_bound=lower_bound, upper_bound=upper_bound)
        safe_notch(raw, picks, notch)
        safe_resample(raw, resample)
        picks = data_channel_indices(raw)
        if scale_uv:
            scale_picks(raw, picks, 1e6 / float(uv_scale))
        return raw

    return pipeline
