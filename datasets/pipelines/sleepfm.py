"""SleepFM Clinical preprocessing preset.

The official pipeline anti-aliases/downsamples to 128 Hz and standardizes each
recording channel. The wrapper retains its five-second (640-sample) tokenizer.
"""

from datasets.pipelines._fm_common import (
    cast_raw_float32,
    data_channel_indices,
    per_channel_zscore,
    safe_resample,
)


def make_pipeline(resample=128.0):
    def pipeline(raw):
        safe_resample(raw, resample)
        picks = data_channel_indices(raw)
        per_channel_zscore(raw, picks)
        return cast_raw_float32(raw)

    return pipeline
