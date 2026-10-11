"""CSBrain-style Raw-level preprocessing preset.

Implements the pretraining signal template: 0.3-75 Hz band-pass, 60 Hz notch,
resampling to 200 Hz, and scaling from volts to units of 100 microvolts.

Omitted by design: fixed-length segmentation and cross-scale tokenization.
Those are dataset epoch/model input responsibilities rather than Raw pipeline
steps.
"""
from datasets.pipelines._fm_common import data_channel_indices, safe_filter, safe_notch, safe_resample, scale_picks


def make_pipeline(
    resample: float = 200.0,
    lower_bound: float = 0.3,
    upper_bound: float = 75.0,
    notch: float = 60.0,
    scale_uv: bool = True,
    uv_scale: float = 100.0,
):
    """Return a ``pipeline(raw) -> raw`` closure for CSBrain."""

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
