"""
models/t_yolov8.py

T-YOLOv8 — Single-stream temporal detection model.

Paper: "Exploiting Temporal Context for Tiny Object Detection"
       Corsel et al., WACVW 2023

Architecture (Section 3.1):
    Input:  X_app ∈ R^{H×W×3}  — 3-channel grayscale temporal stack
            [f_{t-s}, f_t, f_{t+s}]  (one grayscale channel per frame)

    Model:  Standard YOLOv8 backbone + FPN neck + 3 detection heads
            (P3/8, P4/16, P5/32).  The temporal context is encoded solely
            in the 3-channel input; no other architectural changes.

    This class is a thin wrapper around ultralytics DetectionModel that:
      - enforces ch=3 (grayscale triplet)
      - enforces the VisDrone class count (nc=10 by default)
      - provides a named constructor that accepts the t_yolov8.yaml config
      - exposes load_pretrained() to initialise from COCO weights

Usage:
    model = TYOLOv8(scale='x')          # build from scratch
    model = TYOLOv8.from_pretrained('x')  # COCO weights → fine-tune
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

# Ultralytics internals
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER

# Path to our YAML config (adjacent configs/ directory)
_HERE = Path(__file__).parent
_CONFIG_DIR = _HERE.parent / "configs"
_T_YAML = _CONFIG_DIR / "t_yolov8.yaml"

# Mapping: scale letter → official COCO-pretrained weight filename
_PRETRAIN_MAP = {
    "n": "yolov8n.pt",
    "s": "yolov8s.pt",
    "m": "yolov8m.pt",
    "l": "yolov8l.pt",
    "x": "yolov8x.pt",
}


class TYOLOv8(DetectionModel):
    """
    T-YOLOv8 single-stream temporal detection model.

    Inherits DetectionModel (ultralytics) unchanged except for the
    default YAML config (t_yolov8.yaml) and ch=3 enforcement.

    Args:
        scale:   Model scale — one of 'n', 's', 'm', 'l', 'x'.
        nc:      Number of classes (default 10 for VisDrone).
        verbose: Print model summary if True.
    """

    def __init__(
        self,
        scale: str = "x",
        nc: int = 10,
        verbose: bool = True,
    ):
        if not _T_YAML.exists():
            raise FileNotFoundError(
                f"T-YOLOv8 config not found: {_T_YAML}\n"
                "Make sure configs/t_yolov8.yaml is present."
            )

        # Build config dict: inject the requested scale letter
        import yaml  # stdlib

        with open(_T_YAML) as f:
            cfg = yaml.safe_load(f)
        cfg["scale"] = scale

        super().__init__(cfg=cfg, ch=3, nc=nc, verbose=verbose)

        # Store for repr / weight loading
        self.scale = scale
        self._nc = nc

        # Attach default training hyperparameters so that v8DetectionLoss can
        # initialise without requiring a full ultralytics Trainer (model.args).
        if not hasattr(self, "args"):
            from ultralytics.utils import DEFAULT_CFG

            self.args = DEFAULT_CFG

    # ------------------------------------------------------------------
    # Pretrained weight helpers
    # ------------------------------------------------------------------

    def load_pretrained(self, weights_path: Optional[str] = None) -> "TYOLOv8":
        """
        Load COCO-pretrained YOLOv8 weights into the appearance stream,
        skipping the first Conv layer (ch 3→3, compatible) and the final
        Detect head (different nc).

        If weights_path is None, ultralytics will auto-download the
        corresponding yolov8{scale}.pt weights.

        Returns self for chaining.
        """
        if weights_path is None:
            weights_path = _PRETRAIN_MAP.get(self.scale, f"yolov8{self.scale}.pt")

        LOGGER.info(f"T-YOLOv8: loading pretrained weights from {weights_path}")

        # Use ultralytics weight-loading utility which handles shape mismatches
        from ultralytics.nn.tasks import load_checkpoint

        pretrained, _ = load_checkpoint(
            weights_path, device="cpu"
        )  # returns (model, ckpt_dict)

        # Transfer matching weights (skip incompatible layers gracefully)
        state_src = pretrained.state_dict()
        state_dst = self.state_dict()

        transfer = {}
        skipped = []
        for k, v in state_src.items():
            if k in state_dst and state_dst[k].shape == v.shape:
                transfer[k] = v
            else:
                skipped.append(k)

        state_dst.update(transfer)
        self.load_state_dict(state_dst)

        if skipped:
            LOGGER.info(
                f"T-YOLOv8: skipped {len(skipped)} layers with shape mismatch "
                f"(expected for Detect head with nc={self._nc})"
            )
        LOGGER.info(
            f"T-YOLOv8: transferred {len(transfer)}/{len(state_src)} weight tensors"
        )
        return self

    # ------------------------------------------------------------------
    # forward — identical to DetectionModel; documented here for clarity
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor | dict, *args, **kwargs):
        """
        Forward pass.

        Training:   x is a dict with keys 'img' (X_app) and 'bboxes'/'cls'.
                    Returns loss dict.
        Inference:  x is a tensor of shape (B, 3, H, W).
                    Returns predictions.
        """
        return super().forward(x, *args, **kwargs)

    def __repr__(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        return f"TYOLOv8(scale={self.scale!r}, nc={self._nc}, params={total:,})"
