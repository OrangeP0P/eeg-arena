"""SleepLM preprocessing preset: 64 Hz and per-channel z-score."""

from datasets.pipelines._fm_common import (
    cast_raw_float32,
    data_channel_indices,
    per_channel_zscore,
    safe_resample,
)


def make_pipeline(resample=64.0):
    def pipeline(raw):
        safe_resample(raw, resample)
        picks = data_channel_indices(raw)
        per_channel_zscore(raw, picks)
        return cast_raw_float32(raw)

    return pipeline
