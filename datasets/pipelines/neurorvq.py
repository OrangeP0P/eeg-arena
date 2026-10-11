"""NeuroRVQ-style Raw-level preprocessing preset.

The shared Raw preprocessing step resamples the signal to 200 Hz.

Omitted by design: 1 s temporal patch construction, tokenizer/RVQ training
logic, and fixed 104-channel vocabulary enforcement. Those are model input or
data/channels.py responsibilities.
"""
from datasets.pipelines._fm_common import safe_resample


def make_pipeline(resample: float = 200.0):
    """Return a ``pipeline(raw) -> raw`` closure for NeuroRVQ."""

    def pipeline(raw):
        safe_resample(raw, resample)
        return raw

    return pipeline
