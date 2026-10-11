"""Helpers for working with argparse CLI argv."""


def argv_contains_option(argv, option_name):
    """Return True when a long option is explicitly present in argv."""
    prefix = f"{option_name}="
    return any(arg == option_name or arg.startswith(prefix) for arg in (argv or []))


__all__ = ["argv_contains_option"]
