"""
utils/weights.py

Shared weight-transfer helper used by models/t_yolov8.py and models/t2_yolov8.py.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch.nn as nn
from ultralytics.utils import LOGGER


def transfer_weights(
    src_state: dict,
    dst_model: nn.Module,
    key_map: Optional[Callable[[str], str]] = None,
) -> tuple[int, int]:
    """
    Transfer shape-matched weights from src_state into dst_model.

    Args:
        src_state:  Source state dict (e.g. from a pretrained checkpoint).
        dst_model:  Target nn.Module to load weights into.
        key_map:    Optional callable(str) -> str that remaps source keys
                    before matching against the destination state dict.
                    Use this when source and destination have different key
                    naming conventions (e.g. "model.0.conv" -> "layers.0.conv").

    Returns:
        (n_transferred, n_total) — number of transferred tensors and total
        tensors in src_state.
    """
    dst_state = dst_model.state_dict()
    transfer: dict = {}
    skipped: list[str] = []

    for k, v in src_state.items():
        dst_key = key_map(k) if key_map is not None else k
        if dst_key in dst_state and dst_state[dst_key].shape == v.shape:
            transfer[dst_key] = v
        else:
            skipped.append(k)

    dst_state.update(transfer)
    dst_model.load_state_dict(dst_state)

    if skipped:
        LOGGER.info(
            f"transfer_weights: skipped {len(skipped)} tensors "
            "(shape mismatch or key not found)"
        )
    LOGGER.info(
        f"transfer_weights: transferred {len(transfer)}/{len(src_state)} tensors"
    )
    return len(transfer), len(src_state)
