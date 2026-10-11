"""BIOT-style Raw-level preprocessing preset.

Implements resampling to 200 Hz and per-channel normalization by the 95th
percentile of absolute amplitude.

Omitted by design: per-channel tokenization, overlapping token windows, missing
token dropping, and BIOT variant-specific bipolar channel vocabularies. Those
are model/data-layout concerns outside the Raw pipeline.
"""
from datasets.pipelines._fm_common import (
    data_channel_indices,
    per_channel_abs_quantile,
    safe_resample,
)


def make_pipeline(resample: float = 200.0, quantile: float = 0.95):
    """Return a ``pipeline(raw) -> raw`` closure for BIOT."""

    def pipeline(raw):
        safe_resample(raw, resample)
        picks = data_channel_indices(raw)
        per_channel_abs_quantile(raw, picks, quantile=quantile)
        return raw

    return pipeline
