"""
utils/tiling.py

Tile-grid computation and label-clipping helpers for training-time tiling
(paper §3.3: 320×320 crops with 5% overlap).

These are extracted from utils/temporal_dataset.py so they can be imported
and tested independently.
"""

from __future__ import annotations

import numpy as np


def _compute_tile_positions(
    img_h: int,
    img_w: int,
    tile_h: int,
    tile_w: int,
    overlap: float,
) -> list[tuple[int, int, int, int]]:
    """
    Compute a grid of (y0, x0, y1, x1) tile positions for an image of size
    (img_h, img_w) using tiles of size (tile_h, tile_w) with fractional overlap.

    Tiles at the right/bottom boundary are shifted inward so they are always
    exactly (tile_h × tile_w) — no partial tiles.

    Returns:
        List of (y0, x0, y1, x1) tuples in pixel coordinates.
    """
    stride_y = max(1, int(tile_h * (1.0 - overlap)))
    stride_x = max(1, int(tile_w * (1.0 - overlap)))

    tiles: list[tuple[int, int, int, int]] = []

    y0 = 0
    while True:
        y1 = y0 + tile_h
        if y1 > img_h:
            y0 = img_h - tile_h
            y1 = img_h

        x0 = 0
        while True:
            x1 = x0 + tile_w
            if x1 > img_w:
                x0 = img_w - tile_w
                x1 = img_w
            tiles.append((y0, x0, y1, x1))
            if x1 >= img_w:
                break
            x0 += stride_x

        if y1 >= img_h:
            break
        y0 += stride_y

    # Deduplicate (can happen when image is smaller than tile)
    seen: set[tuple[int, int, int, int]] = set()
    unique: list[tuple[int, int, int, int]] = []
    for t in tiles:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def _clip_labels_to_tile(
    labels: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    img_w: int,
    img_h: int,
    min_visibility: float = 0.3,
) -> np.ndarray:
    """
    Transform YOLO-format labels (cls, cx, cy, w, h) normalised to the full
    image into labels normalised to the tile region [x0:x1, y0:y1].

    Boxes that overlap the tile by less than `min_visibility` of their area
    are discarded.

    Args:
        labels: (N, 5) float32 — [cls, cx, cy, w, h] normalised to full image.
        x0, y0, x1, y1: Tile boundaries in pixels (full-image coords).
        img_w, img_h:   Full image dimensions.
        min_visibility: Minimum fraction of box area that must fall inside the
                        tile to keep the box.

    Returns:
        (M, 5) float32 array normalised to the tile, M ≤ N.
    """
    if len(labels) == 0:
        return labels

    tile_w = x1 - x0
    tile_h = y1 - y0

    # Convert normalised cx/cy/w/h → pixel xyxy in full-image coords
    cx_px = labels[:, 1] * img_w
    cy_px = labels[:, 2] * img_h
    bw_px = labels[:, 3] * img_w
    bh_px = labels[:, 4] * img_h

    bx1 = cx_px - bw_px / 2
    by1 = cy_px - bh_px / 2
    bx2 = cx_px + bw_px / 2
    by2 = cy_px + bh_px / 2

    # Clip box to tile
    cx1 = np.clip(bx1, x0, x1)
    cy1 = np.clip(by1, y0, y1)
    cx2 = np.clip(bx2, x0, x1)
    cy2 = np.clip(by2, y0, y1)

    inter_w = np.maximum(0.0, cx2 - cx1)
    inter_h = np.maximum(0.0, cy2 - cy1)
    inter_area = inter_w * inter_h
    box_area = bw_px * bh_px

    # Keep boxes with sufficient visibility (avoid div-by-zero)
    visibility = inter_area / np.maximum(box_area, 1.0)
    keep = visibility >= min_visibility

    if not keep.any():
        return np.zeros((0, 5), dtype=np.float32)

    # Re-normalise to tile coords
    new_cx = (np.clip(cx_px[keep], x0, x1) - x0) / tile_w
    new_cy = (np.clip(cy_px[keep], y0, y1) - y0) / tile_h
    new_bw = np.clip(bw_px[keep], 0, tile_w) / tile_w
    new_bh = np.clip(bh_px[keep], 0, tile_h) / tile_h

    # Clip to [0, 1]
    new_cx = np.clip(new_cx, 0.0, 1.0)
    new_cy = np.clip(new_cy, 0.0, 1.0)
    new_bw = np.clip(new_bw, 0.0, 1.0)
    new_bh = np.clip(new_bh, 0.0, 1.0)

    # Drop degenerate boxes
    valid = (new_bw > 1e-4) & (new_bh > 1e-4)
    cls = labels[keep, 0:1][valid]
    result = np.concatenate(
        [
            cls,
            new_cx[valid, None],
            new_cy[valid, None],
            new_bw[valid, None],
            new_bh[valid, None],
        ],
        axis=1,
    )
    return result.astype(np.float32)
