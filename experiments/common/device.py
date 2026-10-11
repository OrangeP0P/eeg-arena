import torch


def resolve_device(device_arg, cuda_index, gpu_ids_str=None, use_dataparallel=False):
    """Resolve runtime torch.device from CLI args."""
    requested = (device_arg or "auto").lower()

    # Handle multi-GPU with DataParallel
    if use_dataparallel and gpu_ids_str:
        gpu_ids = [int(x.strip()) for x in gpu_ids_str.split(",")]
        if requested == "auto":
            if torch.cuda.is_available() and len(gpu_ids) > 0:
                return torch.device(f"cuda:{gpu_ids[0]}")
            return torch.device("cpu")
        # For DataParallel, use first GPU as primary device
        if requested in ("cuda", "cuda:0"):
            if not torch.cuda.is_available():
                raise ValueError("CUDA requested but no GPU is available.")
            return torch.device(f"cuda:{gpu_ids[0]}")
        if requested.startswith("cuda:"):
            if not torch.cuda.is_available():
                raise ValueError("CUDA requested but no GPU is available.")
            return torch.device(requested)
        raise ValueError(f"Unsupported --device '{device_arg}'. Use auto/cpu/cuda/cuda:N.")

    # Original single-GPU logic
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device(f"cuda:{cuda_index}")
        return torch.device("cpu")

    if requested == "cpu":
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but no GPU is available.")
        return torch.device(f"cuda:{cuda_index}")

    if requested.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but no GPU is available.")
        return torch.device(requested)

    raise ValueError(f"Unsupported --device '{device_arg}'. Use auto/cpu/cuda/cuda:N.")


__all__ = ["resolve_device"]
