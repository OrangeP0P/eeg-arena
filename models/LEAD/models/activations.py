"""Activation helpers used by LEAD."""

import torch.nn.functional as F


def swiglu(x):
    """SwiGLU over the channel dimension of a Conv1d output."""
    x1, x2 = x.chunk(2, dim=1)
    return x1 * F.silu(x2)
