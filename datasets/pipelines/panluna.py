"""PanLUNA EEG preprocessing from arXiv:2604.04297."""
from datasets.pipelines._fm_common import (
    data_channel_indices,
    per_channel_zscore,
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
    """Apply PanLUNA's fourth-order Butterworth EEG path and z-score."""

    def pipeline(raw):
        picks = data_channel_indices(raw)
        if picks:
            sfreq = float(raw.info["sfreq"])
            high = min(float(upper_bound), sfreq / 2.0 - 1e-6)
            if high > float(lower_bound):
                raw.filter(
                    l_freq=float(lower_bound),
                    h_freq=high,
                    picks=picks,
                    method="iir",
                    iir_params={"order": 4, "ftype": "butter"},
                    verbose=False,
                )
        safe_notch(raw, picks, notch)
        safe_resample(raw, resample)
        picks = data_channel_indices(raw)
        if normalize:
            per_channel_zscore(raw, picks)
        return raw

    return pipeline
