"""BrainOmni preprocessing from the official paper/repository."""

from datasets.pipelines._fm_common import (
    cast_raw_float32,
    data_channel_indices,
    safe_filter,
    safe_notch,
    safe_resample,
)


def make_pipeline(
    resample: float = 256.0,
    lower_bound: float = 0.1,
    upper_bound: float = 96.0,
    notch="50,60",
):
    """Notch, resample, and band-pass; segment normalization is in the wrapper."""

    def pipeline(raw):
        picks = data_channel_indices(raw)
        safe_notch(raw, picks, notch)
        safe_resample(raw, resample)
        picks = data_channel_indices(raw)
        safe_filter(raw, picks, lower_bound=lower_bound, upper_bound=upper_bound)
        cast_raw_float32(raw)
        return raw

    return pipeline
