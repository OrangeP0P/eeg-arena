"""SleepGPT preprocessing preset.

The released preprocessing resamples PSG to 100 Hz, applies a 0.3--35 Hz
band-pass, and normalizes channel amplitudes before constructing 30-second
time/frequency inputs. Epoch padding and the STFT remain in the model wrapper.
"""

from datasets.pipelines._fm_common import (
    cast_raw_float32,
    data_channel_indices,
    per_channel_zscore,
    safe_filter,
    safe_resample,
)


def make_pipeline(resample=100.0, lower_bound=0.3, upper_bound=35.0):
    def pipeline(raw):
        picks = data_channel_indices(raw)
        safe_filter(raw, picks, lower_bound=lower_bound, upper_bound=upper_bound)
        safe_resample(raw, resample)
        picks = data_channel_indices(raw)
        per_channel_zscore(raw, picks)
        return cast_raw_float32(raw)

    return pipeline
