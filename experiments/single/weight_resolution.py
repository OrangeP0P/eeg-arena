import os


def handle_list_variants(params, model_config):
    """
    Print available weight variants for --list_variants and indicate if handled.

    Returns:
        True when the request was handled and caller should exit early.
        False when --list_variants is not set and normal flow should continue.
    """
    if not getattr(params, "list_variants", False):
        return False

    weight_variants = getattr(model_config, "WEIGHT_VARIANTS", None)
    if weight_variants:
        default_variant = getattr(model_config, "DEFAULT_VARIANT", "")
        print(f"\nAvailable weight variants for '{params.model}':")
        for name, info in weight_variants.items():
            default_mark = " (default)" if name == default_variant else ""
            description = info.get("description", "")
            print(f"  {name}: {description}{default_mark}")
    else:
        print(
            f"\nModel '{params.model}' has a single weight variant "
            "(use --foundation_dir to override)."
        )
    return True


def resolve_weight_path(params, model_config, parser):
    """
    Resolve pretrained weight path based on --variant.

    Precedence: explicit --foundation_dir > --variant > existing default.
    """
    variant = getattr(params, "variant", None)
    if variant is None:
        return

    weight_variants = getattr(model_config, "WEIGHT_VARIANTS", None)
    if weight_variants is None:
        print("Warning: --variant specified but model has no weight variants. Ignoring.")
        return

    if variant not in weight_variants:
        available = ", ".join(weight_variants.keys())
        raise ValueError(f"Unknown variant '{variant}'. Available: {available}")

    # Check if user explicitly passed --foundation_dir (not just the default)
    foundation_dir_default = parser.get_default("foundation_dir")
    if params.foundation_dir != foundation_dir_default:
        print(f"[Weight Resolver] --foundation_dir explicitly set, ignoring --variant '{variant}'")
        return

    # Construct path from variant
    weights_dir = getattr(model_config, "WEIGHTS_DIR", os.path.dirname(params.foundation_dir))
    weight_file = weight_variants[variant]["file"]
    params.foundation_dir = os.path.join(weights_dir, weight_file)
    print(f"[Weight Resolver] Using variant '{variant}': {params.foundation_dir}")
