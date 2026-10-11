"""mdJPT-style Raw-level preprocessing preset.

Implements common average reference, downsampling to 125 Hz, and 0.5-47 Hz
band-pass filtering.

Omitted by design: 60-channel interpolation, median-threshold artifact-channel
detection, ICA/EEGLAB automatic component removal, and 5 s/2 s windowing.
Those require data-layer/channel-space support rather than a simple Raw
pipeline.
"""
from datasets.pipelines._fm_common import data_channel_indices, safe_filter, safe_resample, set_average_reference


def make_pipeline(
    resample: float = 125.0,
    lower_bound: float = 0.5,
    upper_bound: float = 47.0,
):
    """Return a ``pipeline(raw) -> raw`` closure for mdJPT."""

    def pipeline(raw):
        set_average_reference(raw)
        safe_resample(raw, resample)
        picks = data_channel_indices(raw)
        safe_filter(raw, picks, lower_bound=lower_bound, upper_bound=upper_bound)
        return raw

    return pipeline
