"""Uni-DMFM downstream EEG preprocessing preset."""
from datasets.pipelines._fm_common import (
    data_channel_indices,
    per_channel_zscore,
    safe_filter,
    safe_notch,
    safe_resample,
)


def make_pipeline(
    resample: float = 200.0,
    lower_bound: float = 0.1,
    upper_bound: float = 75.0,
    notch: float = 50.0,
    normalize: bool = True,
):
    """Use the official BCI preprocessing cutoffs and default z-score mode.

    Uni-DMFM's original loader uses training-set channel means/stds from its JSON
    metadata. EEG-Arena cannot know those before its own split, so this leakage-free
    pipeline computes the corresponding statistics within each recording.
    """

    def pipeline(raw):
        picks = data_channel_indices(raw)
        safe_filter(raw, picks, lower_bound=lower_bound, upper_bound=upper_bound)
        safe_notch(raw, picks, notch)
        raw = safe_resample(raw, resample)
        picks = data_channel_indices(raw)
        if normalize:
            per_channel_zscore(raw, picks)
        return raw

    return pipeline
