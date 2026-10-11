import sys

from experiments.common.argparse_utils import argv_contains_option


def apply_model_input_constraints(parser, params, model_config, argv=None):
    """
    Apply model-declared input constraints after argparse parsing.

    Rules:
    - If the model requires a specific input_format / patch_size and the user
      did not pass that flag explicitly, fill in the required value.
    - If the user explicitly passed a conflicting value, fail fast.
    """
    if not hasattr(model_config, "get_input_constraints"):
        return params

    constraints = model_config.get_input_constraints() or {}
    if not constraints:
        return params

    argv = list(sys.argv[1:] if argv is None else argv)

    required_input_format = constraints.get("required_input_format")
    if required_input_format is not None:
        input_format_explicit = argv_contains_option(argv, "--input_format")
        current_input_format = getattr(params, "input_format", None)
        if input_format_explicit and current_input_format != required_input_format:
            parser.error(
                f"Model '{params.model}' requires --input_format {required_input_format}, "
                f"but got '{current_input_format}'."
            )
        if not input_format_explicit:
            params.input_format = required_input_format

    required_patch_size = constraints.get("required_patch_size")
    if required_patch_size is not None:
        patch_size_explicit = argv_contains_option(argv, "--patch_size")
        current_patch_size = getattr(params, "patch_size", None)
        if patch_size_explicit and current_patch_size != required_patch_size:
            parser.error(
                f"Model '{params.model}' requires --patch_size {required_patch_size}, "
                f"but got '{current_patch_size}'."
            )
        if not patch_size_explicit:
            params.patch_size = required_patch_size

    return params


__all__ = ["apply_model_input_constraints"]
