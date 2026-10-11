"""LuMamba preprocessing from arXiv:2603.19100 and BioFoundation."""
from datasets.pipelines._fm_common import (
    data_channel_indices,
    per_channel_zscore,
    safe_filter,
    safe_notch,
    safe_resample,
)


def make_pipeline(
    resample: float = 256.0,
    lower_bound: float = 0.1,
    upper_bound: float = 75.0,
    notch: float = 50.0,
    normalize: bool = True,
):
    """Band-pass, notch, resample and channel-wise normalize LuMamba input.

    The paper uses 50 or 60 Hz notch according to the acquisition site. EEG-Arena
    defaults to 50 Hz for BCIC IV 2b and leaves it CLI-configurable.
    """

    def pipeline(raw):
        picks = data_channel_indices(raw)
        safe_filter(raw, picks, lower_bound=lower_bound, upper_bound=upper_bound)
        safe_notch(raw, picks, notch)
        # The released LuMamba protocol downsamples recordings above 256 Hz,
        # but explicitly leaves lower-rate recordings unchanged.  In
        # particular, BCIC IV 2b therefore remains at its native 250 Hz.
        if resample is not None and float(raw.info["sfreq"]) > float(resample):
            safe_resample(raw, resample)
        picks = data_channel_indices(raw)
        if normalize:
            per_channel_zscore(raw, picks)
        return raw

    return pipeline
