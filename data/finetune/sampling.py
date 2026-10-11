import hashlib
import math

import numpy as np


def validate_data_ratio(value) -> float:
    """Return a validated data sampling ratio in (0, 1]."""
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"data_ratio must be a float in (0, 1], got {value!r}.")

    if not math.isfinite(ratio) or ratio <= 0.0 or ratio > 1.0:
        raise ValueError(f"data_ratio must be in (0, 1], got {ratio}.")
    return ratio


def _stable_sampling_seed(*parts) -> int:
    payload = "::".join(str(part) for part in parts)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2 ** 32)


def sample_index_list_by_ratio(index_list, data_ratio, *, seed=0, split_name="split", context_parts=()):
    """Sample one split deterministically without replacement.

    Sampling is applied over the positional order of ``index_list`` so the
    selected subset matches the historical "Subset of an existing split" logic.
    The returned values are original dataset indices sorted by their original
    split order.
    """
    ratio = validate_data_ratio(data_ratio)
    base_indices = list(index_list)
    original_count = len(base_indices)
    if original_count <= 0:
        raise ValueError(f"Cannot sample data split '{split_name}' because it is empty.")

    if ratio == 1.0:
        return base_indices, {"original": original_count, "sampled": original_count}

    keep_count = min(original_count, max(1, math.floor(original_count * ratio)))
    rng_seed = _stable_sampling_seed(*context_parts, split_name, seed, f"{ratio:.12g}")
    rng = np.random.default_rng(rng_seed)
    selected_positions = sorted(rng.permutation(original_count)[:keep_count].tolist())
    if not selected_positions:
        raise ValueError(
            f"data_ratio={ratio} produced an empty '{split_name}' split from {original_count} samples."
        )

    sampled_indices = [base_indices[position] for position in selected_positions]
    return sampled_indices, {"original": original_count, "sampled": len(sampled_indices)}
