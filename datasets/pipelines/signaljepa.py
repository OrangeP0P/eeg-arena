"""Signal-JEPA preprocessing from the official release and paper."""

from datasets.pipelines._fm_common import (
    cast_raw_float32,
    data_channel_indices,
    safe_filter,
    safe_resample,
    scale_picks,
)


def make_pipeline(
    resample: float = 128.0,
    lower_bound: float = 0.5,
    upper_bound: float = 40.0,
):
    """Band-pass, resample, and convert volts to microvolts."""

    def pipeline(raw):
        picks = data_channel_indices(raw)
        safe_filter(raw, picks, lower_bound=lower_bound, upper_bound=upper_bound)
        safe_resample(raw, resample)
        picks = data_channel_indices(raw)
        scale_picks(raw, picks, 1e6)
        cast_raw_float32(raw)
        return raw

    return pipeline
