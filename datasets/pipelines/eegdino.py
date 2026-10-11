"""EEG-DINO preprocessing following its CBraMod-derived 200-Hz protocol."""
from datasets.pipelines._fm_common import (
    data_channel_indices,
    safe_filter,
    safe_notch,
    safe_resample,
    scale_picks,
)


def make_pipeline(
    resample: float = 200.0,
    lower_bound: float = 0.3,
    upper_bound: float = 75.0,
    notch: float = 50.0,
    scale_01mv: bool = True,
):
    """Prepare EEG-DINO patches in the 0.1-mV units used by its source protocol.

    EEG-DINO states that it follows CBraMod preprocessing: 19 common 10-20
    channels, 200 Hz and 30-second pretraining epochs. CBraMod additionally uses
    0.3--75 Hz filtering, site-specific 50/60 Hz notch, and microvolts divided by
    100. EEG-Arena defaults to 50 Hz for BCIC IV 2b.
    """

    def pipeline(raw):
        picks = data_channel_indices(raw)
        safe_filter(raw, picks, lower_bound=lower_bound, upper_bound=upper_bound)
        safe_notch(raw, picks, notch)
        raw = safe_resample(raw, resample)
        if scale_01mv:
            # MNE stores volts: V * 1e6 / 100 == V * 1e4.
            scale_picks(raw, data_channel_indices(raw), 1e4)
        return raw

    return pipeline
