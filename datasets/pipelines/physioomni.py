"""PhysioOmni EEG preprocessing reproduced from the official dataset scripts."""
from datasets.pipelines._fm_common import (
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
    notch: float = 50.0,
    scale_01mv: bool = True,
):
    """Band-pass/notch/resample EEG and reproduce ``microvolts / 100`` scaling.

    The release uses 50 or 60 Hz according to acquisition site. EEG-Arena uses 50 Hz
    by default for BCIC IV 2b and exposes ``notch`` as a pipeline parameter.
    """

    def pipeline(raw):
        picks = data_channel_indices(raw)
        safe_filter(raw, picks, lower_bound=lower_bound, upper_bound=upper_bound)
        safe_notch(raw, picks, notch)
        raw = safe_resample(raw, resample)
        if scale_01mv:
            scale_picks(raw, data_channel_indices(raw), 1e4)
        return raw

    return pipeline
