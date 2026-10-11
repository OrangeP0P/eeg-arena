"""Dataset-specific OpenNeuro kernels package.

Each module in this package provides a ``KERNEL`` object for one OpenNeuro
dataset.  Modules are named ``openneuro_{ds_id}.py`` (e.g.
``openneuro_ds004444.py``).

The dispatch logic in stage-2 extraction should:
    1. Derive ds_id from the dataset path (e.g. ``ds004444``).
    2. Try ``from kernels.openneuro_specific import openneuro_{ds_id}; kernel = openneuro_{ds_id}.KERNEL``.
    3. Fall back to the generic BIDS kernel if the module does not exist.
"""
