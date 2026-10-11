"""
Random seed utilities for reproducible training
"""
import random
import numpy as np
import torch


def setup_seed(seed):
    """
    Set random seed for reproducible training

    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
