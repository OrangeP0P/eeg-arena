"""HEAR-style Raw-level preprocessing preset.

The integrated HEAR checkpoints use 200 Hz patched input with 1 s patches
(``patch_size=200``). The upstream fine-tuning scripts operate on 800-sample
windows and do not apply trial-level z-score normalization in the model code.

This preset therefore keeps the Raw-level steps conservative: 0.1-75 Hz
band-pass, configurable power-line notch, resampling to 200 Hz, and optional
scaling from volts to units of 100 microvolts.

Omitted by design: the 800-sample / 4-patch epoch window and fixed channel-space
selection. Those are handled by task epoch settings, patched input formatting,
and ``data/channels.py``.
"""
from datasets.pipelines._fm_common import data_channel_indices, safe_filter, safe_notch, safe_resample, scale_picks


def make_pipeline(
    resample: float = 200.0,
    lower_bound: float = 0.1,
    upper_bound: float = 75.0,
    notch: float = 50.0,
    scale_uv: bool = True,
    uv_scale: float = 100.0,
):
    """Return a ``pipeline(raw) -> raw`` closure for HEAR."""

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
