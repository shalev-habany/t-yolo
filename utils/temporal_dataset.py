"""
utils/temporal_dataset.py

TemporalDataset — PyTorch Dataset for temporal tiny object detection.

For each annotated key-frame f_t, retrieves the triplet:
    (f_{t-s},  f_t,  f_{t+s})

and produces two inputs:

    X_app  ∈ R^{3 × H × W}   — grayscale temporal stack (appearance stream)
    X_mot  ∈ R^{2 × H × W}   — absolute frame difference (motion stream)

Annotation labels belong to f_t only (support frames are unlabelled, per paper).

Training tiling (paper §3.3):
    At training time the dataset tiles each full-resolution triplet into
    multiple 320×320 crops with 5% overlap.  All three frames in the triplet
    are cropped at exactly the same (x, y) position so that temporal
    consistency is preserved.  Only the tiles that contain at least one GT
    box (after clipping) are kept.  At inference the full resolution frame is
    used (img_size controls the resize target).

Sequence folder structure expected:
    <split_root>/
        <seq_id>/
            frames/
                frame_000001.jpg
                frame_000002.jpg
                ...
            labels/
                frame_000001.txt   (YOLO format, for key-frames only)
                ...

The dataset indexes all frames that have a corresponding label file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Protocol, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class _Registrar(Protocol):
    def register(self, src: np.ndarray, ref: np.ndarray) -> np.ndarray: ...


class _Augmentor(Protocol):
    def __call__(
        self,
        f_pre: np.ndarray,
        f_key: np.ndarray,
        f_post: np.ndarray,
        labels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...


class TemporalDataset(Dataset):
    """
    Temporal triplet dataset for T-YOLOv8 / T2-YOLOv8.

    Args:
        split_root:      Path to the split directory (e.g. data/VisDrone/sequences/train).
        temporal_shift:  Frame index offset s. Triplet = (t-s, t, t+s).
        img_size:        (H, W) used at inference (full-resolution resize target).
                         Ignored at training time when tiling is enabled.
        tile_size:       (H, W) of each training tile crop. Paper uses (320, 320).
                         Set to None to disable tiling (resize to img_size instead).
        tile_overlap:    Fractional overlap between adjacent tiles (0.05 = 5%).
        register_frames: If True, align support frames onto f_t before stacking
                         (required for moving-camera datasets like VisDrone).
        registration_method: 'ecc' (fast) or 'sift' (robust). Only used if
                             register_frames=True.
        two_stream:      If True, also returns X_mot (motion stream input).
                         Set to False for single-stream T-YOLOv8.
        augment:         If True, applies TemporalAugmentation.
        augmentor:       Optional TemporalAugmentor instance. If None and
                         augment=True, a default one is created.
    """

    def __init__(
        self,
        split_root: str | Path,
        temporal_shift: int = 3,
        img_size: tuple[int, int] = (640, 640),
        tile_size: Optional[tuple[int, int]] = None,
        tile_overlap: float = 0.05,
        register_frames: bool = True,
        registration_method: str = "ecc",
        two_stream: bool = True,
        augment: bool = False,
        augmentor: Optional[_Augmentor] = None,
    ):
        self.split_root = Path(split_root)
        self.temporal_shift = temporal_shift
        self.img_size = img_size  # (H, W) — inference resize target
        self.tile_size = tile_size  # (H, W) — training tile size; None = no tiling
        self.tile_overlap = tile_overlap
        self.register_frames = register_frames
        self.two_stream = two_stream
        self.augment = augment

        # Typed as Optional[_Registrar] / Optional[_Augmentor] so the Protocol
        # methods are visible to the type checker below.
        self.registrar: Optional[_Registrar]
        self.augmentor: Optional[_Augmentor]

        # Build frame registrar
        if register_frames:
            from utils.frame_registration import build_registrar

            self.registrar = build_registrar(registration_method)  # type: ignore[assignment]
        else:
            self.registrar = None

        # Build augmentor
        if augment:
            if augmentor is not None:
                self.augmentor = augmentor
            else:
                from utils.temporal_augmentation import TemporalAugmentor

                self.augmentor = TemporalAugmentor(img_size=tile_size or img_size)  # type: ignore[assignment]
        else:
            self.augmentor = None

        # Index all labelled key-frames across all sequences.
        # When tiling is enabled the sample list is expanded to one entry per
        # (frame-triplet, tile-position) pair.
        self.samples: list[dict] = []  # each entry has seq_dir, frame_paths, label_path
        # When tiling: also includes tile crop coordinates (x0, y0, x1, y1)
        self._index_sequences()

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def _index_sequences(self) -> None:
        """Walk split_root and collect all labelled frames per sequence.

        When tiling is enabled each frame entry is expanded to one sample
        per tile position so that __getitem__ can work with a flat index.
        """
        if not self.split_root.exists():
            raise FileNotFoundError(f"Split root not found: {self.split_root}")

        for seq_dir in sorted(self.split_root.iterdir()):
            if not seq_dir.is_dir():
                continue

            frames_dir = seq_dir / "frames"
            labels_dir = seq_dir / "labels"

            if not frames_dir.exists():
                continue

            # Sorted list of ALL frame paths in this sequence
            all_frames = sorted(
                list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png"))
            )
            if not all_frames:
                continue

            for i, frame_path in enumerate(all_frames):
                # Only index frames that have a label file
                label_path = labels_dir / (frame_path.stem + ".txt")
                if not label_path.exists():
                    continue

                # Resolve support frame indices (clamp to sequence boundaries)
                i_pre = max(0, i - self.temporal_shift)
                i_post = min(len(all_frames) - 1, i + self.temporal_shift)

                base_entry = {
                    "key_path": frame_path,
                    "pre_path": all_frames[i_pre],
                    "post_path": all_frames[i_post],
                    "label_path": label_path,
                    "seq_id": seq_dir.name,
                }

                if self.tile_size is not None:
                    # Compute tile grid from the natural image size.
                    # We peek at image shape via OpenCV (header only).
                    img = cv2.imread(str(frame_path))
                    if img is None:
                        continue
                    orig_h, orig_w = img.shape[:2]
                    tiles = _compute_tile_positions(
                        orig_h,
                        orig_w,
                        self.tile_size[0],
                        self.tile_size[1],
                        self.tile_overlap,
                    )
                    for y0, x0, y1, x1 in tiles:
                        entry = dict(base_entry)
                        entry["tile"] = (y0, x0, y1, x1)
                        self.samples.append(entry)
                else:
                    self.samples.append(base_entry)

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No labelled frames found under {self.split_root}. "
                "Check that labels/ directories contain .txt files."
            )

    # ------------------------------------------------------------------
    # Core loading
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # Load the three frames
        f_pre = _load_gray(sample["pre_path"])
        f_key = _load_gray(sample["key_path"])
        f_post = _load_gray(sample["post_path"])

        # Register support frames onto key frame (remove camera motion)
        if self.registrar is not None:
            f_pre = self.registrar.register(f_pre, f_key)
            f_post = self.registrar.register(f_post, f_key)

        # Load labels (belong to f_key only)
        labels = _load_labels(sample["label_path"])

        # --- Training tile crop (paper §3.3) ---
        # All three frames are cropped at the SAME position to preserve
        # temporal consistency.  Labels are clipped/filtered to the tile.
        if "tile" in sample:
            y0, x0, y1, x1 = sample["tile"]
            orig_h, orig_w = f_key.shape[:2]
            f_pre = f_pre[y0:y1, x0:x1]
            f_key = f_key[y0:y1, x0:x1]
            f_post = f_post[y0:y1, x0:x1]
            labels = _clip_labels_to_tile(labels, x0, y0, x1, y1, orig_w, orig_h)

        # Apply temporal augmentation (before final resize, in pixel space)
        if self.augmentor is not None:
            f_pre, f_key, f_post, labels = self.augmentor(f_pre, f_key, f_post, labels)

        # Determine output size
        if "tile" in sample and self.tile_size is not None:
            out_H, out_W = self.tile_size
        else:
            out_H, out_W = self.img_size

        # Resize all frames to target size
        f_pre = cv2.resize(f_pre, (out_W, out_H), interpolation=cv2.INTER_LINEAR)
        f_key = cv2.resize(f_key, (out_W, out_H), interpolation=cv2.INTER_LINEAR)
        f_post = cv2.resize(f_post, (out_W, out_H), interpolation=cv2.INTER_LINEAR)

        # --- Build X_app: 3-channel grayscale stack (appearance stream) ---
        # Shape: (3, H, W), float32 in [0, 1]
        X_app = np.stack([f_pre, f_key, f_post], axis=0).astype(np.float32) / 255.0
        X_app_t = torch.from_numpy(X_app)

        result = {
            "X_app": X_app_t,  # (3, H, W)
            "labels": labels,  # (N, 5) — [cls, cx, cy, w, h]
            "key_path": str(sample["key_path"]),
            "seq_id": sample["seq_id"],
        }

        # --- Build X_mot: 2-channel absolute difference (motion stream) ---
        if self.two_stream:
            d_pre = (
                np.abs(f_pre.astype(np.int16) - f_key.astype(np.int16))
                .clip(0, 255)
                .astype(np.uint8)
            )
            d_post = (
                np.abs(f_post.astype(np.int16) - f_key.astype(np.int16))
                .clip(0, 255)
                .astype(np.uint8)
            )
            X_mot = np.stack([d_pre, d_post], axis=0).astype(np.float32) / 255.0
            result["X_mot"] = torch.from_numpy(X_mot)  # (2, H, W)

        return result


# ------------------------------------------------------------------
# DataLoader collate function
# ------------------------------------------------------------------


def temporal_collate_fn(batch: list[dict]) -> dict:
    """
    Custom collate: stacks tensors, packs labels with batch index prefix.
    Labels: list of (N_i, 5) -> padded (sum_N, 6) with col-0 = batch_idx.
    """
    X_app = torch.stack([b["X_app"] for b in batch], dim=0)  # (B, 3, H, W)
    has_mot = "X_mot" in batch[0]
    if has_mot:
        X_mot = torch.stack([b["X_mot"] for b in batch], dim=0)  # (B, 2, H, W)

    # Pack labels: prepend batch index
    label_list = []
    for bi, b in enumerate(batch):
        lbl = b["labels"]  # (N, 5) numpy or tensor
        if isinstance(lbl, np.ndarray):
            lbl = torch.from_numpy(lbl).float()
        if lbl.numel() == 0:
            continue
        idx_col = torch.full((lbl.shape[0], 1), bi, dtype=torch.float32)
        label_list.append(torch.cat([idx_col, lbl], dim=1))  # (N, 6)

    labels = torch.cat(label_list, dim=0) if label_list else torch.zeros((0, 6))

    result = {
        "X_app": X_app,
        "labels": labels,
        "key_paths": [b["key_path"] for b in batch],
    }
    if has_mot:
        result["X_mot"] = X_mot  # type: ignore[possibly-undefined]

    return result


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _load_gray(path: Path) -> np.ndarray:
    """Load an image as a single-channel grayscale uint8 array (H, W)."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load frame: {path}")
    return img


def _load_labels(path: Path) -> np.ndarray:
    """
    Load YOLO-format label file.
    Returns float32 array of shape (N, 5): [cls, cx, cy, w, h].
    Returns empty (0, 5) array if file is empty or missing.
    """
    if not path.exists() or path.stat().st_size == 0:
        return np.zeros((0, 5), dtype=np.float32)

    lines = path.read_text().strip().splitlines()
    rows = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5:
            rows.append([float(p) for p in parts])

    if not rows:
        return np.zeros((0, 5), dtype=np.float32)

    return np.array(rows, dtype=np.float32)


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
        tiles_row_start = len(tiles)

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
