"""OSF-Base preprocessing preset: 64 Hz standardized PSG clipped to [-6, 6]."""

from datasets.pipelines._fm_common import (
    cast_raw_float32,
    clip_picks,
    data_channel_indices,
    per_channel_zscore,
    safe_resample,
)


def make_pipeline(resample=64.0, normalize=True, clip=6.0):
    def pipeline(raw):
        safe_resample(raw, resample)
        picks = data_channel_indices(raw)
        if normalize:
            per_channel_zscore(raw, picks)
        clip_picks(raw, picks, -float(clip), float(clip))
        return cast_raw_float32(raw)

    return pipeline
