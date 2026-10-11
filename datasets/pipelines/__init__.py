"""
EEG Preprocessing Pipeline Library

Each pipeline is a .py file that exports a module-level callable named `pipeline`:
    def pipeline(raw: mne.io.Raw) -> mne.io.Raw

Usage in run_single_experiment.py:
    --pipeline pipelines/default.py      # use built-in preset
    --pipeline /path/to/my_pipeline.py   # use custom pipeline
    (omit --pipeline or pass None)       # no preprocessing

The pipeline file's SHA256 (first 8 hex chars) is embedded in the HDF5 filename
to uniquely identify the preprocessing configuration.
"""
