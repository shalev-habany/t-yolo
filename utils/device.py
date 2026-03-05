"""
utils/device.py

Shared device-resolution helper used by train.py, val.py, and smoke_test.py.
"""

from __future__ import annotations

import torch


def resolve_device(device_str: str = "0") -> torch.device:
    """
    Resolve a device string to a torch.device.

    Args:
        device_str: "cpu", or a CUDA index like "0", "1".

    Returns:
        torch.device — falls back to CPU with a warning if CUDA is not available.
    """
    if device_str == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device(f"cuda:{device_str}")
    print("CUDA not available, falling back to CPU")
    return torch.device("cpu")
