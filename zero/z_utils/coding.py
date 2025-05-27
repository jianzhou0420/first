
import os
import re
from numpy import array as npa
from typing import Dict, Callable
import torch
import numpy as np


def tensorfp32(x):
    if isinstance(x, list):
        return torch.tensor(x, dtype=torch.float32)
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x, dtype=torch.float32)
    else:
        raise TypeError("Input must be a list or numpy array")


def check_and_make(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def dict_apply(
        x: Dict[str, torch.Tensor],
        func: Callable[[torch.Tensor], torch.Tensor]
) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))
