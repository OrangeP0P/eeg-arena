"""TFM-Tokenizer preprocessing from its official dataset loaders and scripts."""
from datasets.pipelines._fm_common import (
    data_channel_indices,
    per_channel_abs_quantile,
    safe_filter,
    safe_notch,
    safe_resample,
)


def make_pipeline(
    resample: float = 200.0,
    lower_bound: float = 0.1,
    upper_bound: float = 75.0,
    notch: float = 50.0,
    quantile: float = 0.95,
    normalize: bool = True,
):
    """Filter at the paper cutoffs and divide each channel by its 95th percentile."""

    def pipeline(raw):
        picks = data_channel_indices(raw)
        safe_filter(raw, picks, lower_bound=lower_bound, upper_bound=upper_bound)
        safe_notch(raw, picks, notch)
        raw = safe_resample(raw, resample)
        picks = data_channel_indices(raw)
        if normalize:
            per_channel_abs_quantile(raw, picks, quantile=quantile)
        return raw

    return pipeline
