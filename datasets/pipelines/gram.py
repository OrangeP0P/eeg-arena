"""Official Gram preprocessing preset."""
from datasets.pipelines._fm_common import (
    data_channel_indices,
    safe_filter,
    safe_notch,
    safe_resample,
    scale_picks,
    set_average_reference,
)


def make_pipeline(
    resample: float = 200.0,
    lower_bound: float = 0.1,
    upper_bound: float = 75.0,
    notch: float = 50.0,
    scale_01mv: bool = True,
):
    """Average-reference, filter, and express Gram input in units of 0.1 mV."""

    def pipeline(raw):
        raw = safe_resample(raw, resample)
        set_average_reference(raw)
        picks = data_channel_indices(raw)
        safe_filter(raw, picks, lower_bound=lower_bound, upper_bound=upper_bound)
        safe_notch(raw, picks, notch)
        if scale_01mv:
            scale_picks(raw, picks, 1e4)
        return raw

    return pipeline
