"""NeuroLM preprocessing from the official paper and repository."""

from datasets.pipelines._fm_common import (
    cast_raw_float32,
    data_channel_indices,
    safe_filter,
    safe_notch,
    safe_resample,
    scale_picks,
)


def make_pipeline(
    resample: float = 200.0,
    lower_bound: float = 0.1,
    upper_bound: float = 75.0,
    notch="50,60",
):
    """Filter to 0.1--75 Hz, notch line noise, resample, and use microvolts."""

    def pipeline(raw):
        picks = data_channel_indices(raw)
        safe_notch(raw, picks, notch)
        safe_filter(raw, picks, lower_bound=lower_bound, upper_bound=upper_bound)
        safe_resample(raw, resample)
        picks = data_channel_indices(raw)
        scale_picks(raw, picks, 1e6)
        cast_raw_float32(raw)
        return raw

    return pipeline
