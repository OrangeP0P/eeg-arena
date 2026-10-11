"""FEMBA preprocessing from arXiv:2502.06438 and BioFoundation.

The released data builder writes microvolt-valued 0.1--75 Hz, 256 Hz EEG.
The released training configuration constructs the TCP bipolar montage and
then applies ``(x - q_lower) / (q_upper - q_lower)``.  That ordered operation
lives in the model wrapper because an MNE ``Raw`` pipeline cannot represent
missing target slots.  This pipeline stops after producing microvolt-valued
monopolar/direct-bipolar signals.
"""
from datasets.pipelines._fm_common import (
    data_channel_indices,
    safe_filter,
    safe_notch,
    safe_resample,
    scale_picks,
)


def make_pipeline(
    resample: float = 256,
    lower_bound: float = 0.1,
    upper_bound: float = 75.0,
    notch: float = 60.0,
    scale_uv: bool = True,
):
    """Return a ``pipeline(raw) -> raw`` closure using released defaults."""
    def pipeline(raw):
        picks = data_channel_indices(raw)
        safe_filter(raw, picks, lower_bound=lower_bound, upper_bound=upper_bound)
        safe_notch(raw, picks, notch)
        safe_resample(raw, resample)
        picks = data_channel_indices(raw)
        if scale_uv:
            scale_picks(raw, picks, 1e6)

        return raw

    return pipeline
