"""Common experiment utilities shared by debug and batch flows."""

from .argparse_utils import argv_contains_option

__all__ = ["ExperimentTracker", "argv_contains_option", "now_iso", "resolve_device", "scalarize_metrics"]


def __getattr__(name):
    if name == "resolve_device":
        from .device import resolve_device

        return resolve_device
    if name in {"ExperimentTracker", "now_iso", "scalarize_metrics"}:
        from .tracking import ExperimentTracker, now_iso, scalarize_metrics

        exported = {
            "ExperimentTracker": ExperimentTracker,
            "now_iso": now_iso,
            "scalarize_metrics": scalarize_metrics,
        }
        return exported[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
