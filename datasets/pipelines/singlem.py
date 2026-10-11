"""Official SingLEM preprocessing preset.

The released pipeline uses 50 Hz notch filtering, 0.5--50 Hz band-pass,
128 Hz resampling, microvolt conversion, and a fixed 0.01 input scale.
"""
from datasets.pipelines._fm_common import (
    cast_raw_float32,
    data_channel_indices,
    safe_filter,
    safe_notch,
    safe_resample,
    scale_picks,
)


def make_pipeline(
    resample: float = 128.0,
    lower_bound: float = 0.5,
    upper_bound: float = 50.0,
    notch: float = 50.0,
    input_scale: float = 0.01,
):
    """Return the preprocessing closure used by the official SingLEM release."""

    def pipeline(raw):
        picks = data_channel_indices(raw)
        safe_notch(raw, picks, notch)
        safe_filter(raw, picks, lower_bound=lower_bound, upper_bound=upper_bound)
        safe_resample(raw, resample)
        picks = data_channel_indices(raw)
        scale_picks(raw, picks, 1e6 * float(input_scale))
        cast_raw_float32(raw)
        return raw

    return pipeline
