"""
utils/temporal_augmentation.py

Temporal Data Augmentation — a key contribution of the paper.

The core rule: every spatial augmentation must be applied with
IDENTICAL parameters to all three frames (f_{t-s}, f_t, f_{t+s}).

Reason: the motion stream depends on |f_{t-s} - f_t| and |f_t - f_{t+s}|.
If frames are augmented independently (standard YOLOv8 augmentation) the
absolute difference will include augmentation artefacts, not real motion,
corrupting the motion signal the model is supposed to learn.

Augmentations implemented:
  - Horizontal flip
  - Vertical flip
  - Random scale + translate (affine)
  - Random rotation
  - HSV jitter  (applied to each frame independently — colour, NOT spatial)
  - Temporal Mosaic (4-tile mosaic built from 4 complete triplets)
  - Temporal MixUp (blend a second triplet at the same alpha)

Usage:
    augmentor = TemporalAugmentor(img_size=(640, 640), mosaic_p=0.5)
    f_pre, f_key, f_post, labels = augmentor(f_pre, f_key, f_post, labels)

All frames are uint8 grayscale (H, W) numpy arrays.
Labels are float32 (N, 5): [cls, cx, cy, w, h] normalised.
"""

from __future__ import annotations

import random
from typing import Callable, Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _value_jitter_delta(frame: np.ndarray, delta: int) -> np.ndarray:
    """
    Apply a fixed brightness delta to a grayscale frame (clipped to [0, 255]).
    Used by TemporalAugmentor to apply the same delta to all frames in a triplet.
    """
    frame = frame.astype(np.int16) + delta
    return np.clip(frame, 0, 255).astype(np.uint8)


def _value_jitter(frame: np.ndarray, magnitude: float) -> np.ndarray:
    """
    Apply random brightness (value channel) jitter to a grayscale frame.
    Magnitude: fraction of 255.  Samples its own delta — use only when
    independent-per-frame jitter is intentional.
    """
    delta = int(random.uniform(-magnitude, magnitude) * 255)
    return _value_jitter_delta(frame, delta)


def _clip_boxes(labels: np.ndarray) -> np.ndarray:
    """Clip normalised box coordinates to [0, 1] and remove degenerate boxes."""
    if len(labels) == 0:
        return labels
    labels[:, 1:5] = np.clip(labels[:, 1:5], 0.0, 1.0)
    # Remove boxes where w or h collapsed to near-zero
    keep = (labels[:, 3] > 1e-3) & (labels[:, 4] > 1e-3)
    return labels[keep]


def _apply_affine_to_labels(
    labels: np.ndarray, M: np.ndarray, orig_w: int, orig_h: int, new_w: int, new_h: int
) -> np.ndarray:
    """
    Apply a 2×3 affine matrix M to YOLO-format bounding boxes.
    Transforms all 4 corners and re-fits axis-aligned bounding boxes.
    """
    if len(labels) == 0:
        return labels

    cls = labels[:, 0:1]
    # Convert cx, cy, w, h -> x1, y1, x2, y2 in pixel coords
    cx = labels[:, 1] * orig_w
    cy = labels[:, 2] * orig_h
    bw = labels[:, 3] * orig_w
    bh = labels[:, 4] * orig_h

    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2

    # All 4 corners
    corners = np.stack(
        [
            np.stack([x1, y1], axis=1),
            np.stack([x2, y1], axis=1),
            np.stack([x1, y2], axis=1),
            np.stack([x2, y2], axis=1),
        ],
        axis=1,
    )  # (N, 4, 2)

    N = len(labels)
    corners_flat = corners.reshape(-1, 2)
    ones = np.ones((corners_flat.shape[0], 1))
    corners_h = np.hstack([corners_flat, ones])  # (4N, 3)
    transformed = (M @ corners_h.T).T  # (4N, 2)
    transformed = transformed.reshape(N, 4, 2)

    x_vals = transformed[:, :, 0]
    y_vals = transformed[:, :, 1]

    new_x1 = x_vals.min(axis=1)
    new_y1 = y_vals.min(axis=1)
    new_x2 = x_vals.max(axis=1)
    new_y2 = y_vals.max(axis=1)

    # Back to YOLO normalised
    new_cx = ((new_x1 + new_x2) / 2) / new_w
    new_cy = ((new_y1 + new_y2) / 2) / new_h
    new_bw = (new_x2 - new_x1) / new_w
    new_bh = (new_y2 - new_y1) / new_h

    new_labels = np.concatenate(
        [
            cls,
            new_cx[:, None],
            new_cy[:, None],
            new_bw[:, None],
            new_bh[:, None],
        ],
        axis=1,
    )
    return _clip_boxes(new_labels)


# ---------------------------------------------------------------------------
# Core augmentor
# ---------------------------------------------------------------------------


class TemporalAugmentor:
    """
    Applies temporally-consistent spatial augmentations to a frame triplet.

    Args:
        img_size:    (H, W) output size.
        hflip_p:     Probability of horizontal flip.
        vflip_p:     Probability of vertical flip.
        scale_range: (min_scale, max_scale) for random scaling.
        translate_frac: Max fraction of image to translate in x/y.
        rotate_deg:  Max rotation in degrees (±).
        hsv_p:       Probability of HSV jitter (applied per-frame independently —
                     colour is not temporal, so independent is correct).
        hsv_h:       HSV hue jitter fraction.
        hsv_s:       HSV saturation jitter fraction.
        hsv_v:       HSV value jitter fraction.
        mosaic_p:    Probability of Temporal Mosaic augmentation.
        mixup_p:     Probability of Temporal MixUp augmentation.
        triplet_provider: Callable that returns a random (f_pre, f_key, f_post, labels)
                          triplet. Required if mosaic_p > 0 or mixup_p > 0.
                          If None, Mosaic/MixUp are disabled regardless of probabilities.
    """

    def __init__(
        self,
        img_size: tuple[int, int] = (640, 640),
        hflip_p: float = 0.5,
        vflip_p: float = 0.0,
        scale_range: tuple[float, float] = (0.5, 1.5),
        translate_frac: float = 0.1,
        rotate_deg: float = 0.0,
        hsv_p: float = 0.5,
        hsv_h: float = 0.015,
        hsv_s: float = 0.7,
        hsv_v: float = 0.4,
        mosaic_p: float = 0.5,
        mixup_p: float = 0.1,
        triplet_provider: Optional[Callable] = None,
    ):
        self.img_size = img_size  # (H, W)
        self.hflip_p = hflip_p
        self.vflip_p = vflip_p
        self.scale_range = scale_range
        self.translate_frac = translate_frac
        self.rotate_deg = rotate_deg
        self.hsv_p = hsv_p
        self.hsv_h = hsv_h
        self.hsv_s = hsv_s
        self.hsv_v = hsv_v
        self.mosaic_p = mosaic_p if triplet_provider is not None else 0.0
        self.mixup_p = mixup_p if triplet_provider is not None else 0.0
        self.triplet_provider = triplet_provider

    def __call__(
        self,
        f_pre: np.ndarray,
        f_key: np.ndarray,
        f_post: np.ndarray,
        labels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Augment a triplet.

        Args:
            f_pre, f_key, f_post: uint8 grayscale (H, W) frames.
            labels: float32 (N, 5) — [cls, cx, cy, w, h] normalised.

        Returns:
            Augmented (f_pre, f_key, f_post, labels).
        """
        H, W = self.img_size

        # 1. Temporal Mosaic (builds a new triplet from 4 random ones)
        if self.mosaic_p > 0 and random.random() < self.mosaic_p:
            f_pre, f_key, f_post, labels = self._temporal_mosaic(
                f_pre, f_key, f_post, labels
            )

        # 2. Temporal MixUp
        if self.mixup_p > 0 and random.random() < self.mixup_p:
            f_pre, f_key, f_post, labels = self._temporal_mixup(
                f_pre, f_key, f_post, labels
            )

        orig_h, orig_w = f_key.shape[:2]

        # 3. Horizontal flip (same decision for all frames)
        if random.random() < self.hflip_p:
            f_pre = np.fliplr(f_pre)
            f_key = np.fliplr(f_key)
            f_post = np.fliplr(f_post)
            if len(labels):
                labels[:, 1] = 1.0 - labels[:, 1]  # cx flipped

        # 4. Vertical flip
        if random.random() < self.vflip_p:
            f_pre = np.flipud(f_pre)
            f_key = np.flipud(f_key)
            f_post = np.flipud(f_post)
            if len(labels):
                labels[:, 2] = 1.0 - labels[:, 2]  # cy flipped

        # 5. Random affine: scale + translate (+ optional rotate)
        #    Single matrix computed ONCE, applied to all 3 frames.
        scale = random.uniform(*self.scale_range)
        tx = random.uniform(-self.translate_frac, self.translate_frac) * orig_w
        ty = random.uniform(-self.translate_frac, self.translate_frac) * orig_h
        angle = random.uniform(-self.rotate_deg, self.rotate_deg)

        M = _build_affine_matrix(scale, tx, ty, angle, orig_w, orig_h, W, H)

        f_pre = cv2.warpAffine(
            f_pre, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        f_key = cv2.warpAffine(
            f_key, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        f_post = cv2.warpAffine(
            f_post, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )

        labels = _apply_affine_to_labels(labels, M, orig_w, orig_h, W, H)

        # 6. HSV jitter — same delta applied to all 3 frames to preserve the
        #    motion signal.  If frames were jittered independently, the absolute
        #    differences |f_pre - f_key| would pick up artefacts from the
        #    brightness shift rather than true object motion.
        if random.random() < self.hsv_p:
            delta = int(random.uniform(-self.hsv_v, self.hsv_v) * 255)
            f_pre = _value_jitter_delta(f_pre, delta)
            f_key = _value_jitter_delta(f_key, delta)
            f_post = _value_jitter_delta(f_post, delta)

        return f_pre, f_key, f_post, labels

    # ------------------------------------------------------------------
    # Temporal Mosaic
    # ------------------------------------------------------------------

    def _temporal_mosaic(
        self,
        f_pre0: np.ndarray,
        f_key0: np.ndarray,
        f_post0: np.ndarray,
        labels0: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Temporal Mosaic Augmentation (Section 3.2 of paper).

        Sample 3 additional random triplets and build a 4-tile mosaic.
        The mosaic is built channel-wise: each tile occupies a quadrant
        in all three frames simultaneously, preserving temporal consistency.

        The same spatial mosaic layout is applied to all 3 channels.
        """
        assert self.triplet_provider is not None
        H, W = self.img_size
        half_h, half_w = H // 2, W // 2

        # Collect 4 triplets (this one + 3 random)
        triplets = [(f_pre0, f_key0, f_post0, labels0)]
        for _ in range(3):
            triplets.append(self.triplet_provider())

        # Mosaic canvas for each frame in the triplet
        canvas_pre = np.zeros((H, W), dtype=np.uint8)
        canvas_key = np.zeros((H, W), dtype=np.uint8)
        canvas_post = np.zeros((H, W), dtype=np.uint8)
        all_labels: list[np.ndarray] = []

        # Quadrant placements: (row_start, col_start, row_end, col_end)
        quads = [
            (0, 0, half_h, half_w),
            (0, half_w, half_h, W),
            (half_h, 0, H, half_w),
            (half_h, half_w, H, W),
        ]

        for (r0, c0, r1, c1), (fp, fk, fpo, lbl) in zip(quads, triplets):
            qh, qw = r1 - r0, c1 - c0
            # Resize each frame in the triplet to the quadrant size
            rp = cv2.resize(fp, (qw, qh), interpolation=cv2.INTER_LINEAR)
            rk = cv2.resize(fk, (qw, qh), interpolation=cv2.INTER_LINEAR)
            rpo = cv2.resize(fpo, (qw, qh), interpolation=cv2.INTER_LINEAR)

            canvas_pre[r0:r1, c0:c1] = rp
            canvas_key[r0:r1, c0:c1] = rk
            canvas_post[r0:r1, c0:c1] = rpo

            # Adjust label coordinates to the mosaic tile
            if len(lbl):
                tile_lbl = lbl.copy()
                orig_h, orig_w = fk.shape[:2]
                # Scale box to original tile size, shift to mosaic position
                tile_lbl[:, 1] = (lbl[:, 1] * orig_w * (qw / orig_w) + c0) / W
                tile_lbl[:, 2] = (lbl[:, 2] * orig_h * (qh / orig_h) + r0) / H
                tile_lbl[:, 3] = lbl[:, 3] * orig_w * (qw / orig_w) / W
                tile_lbl[:, 4] = lbl[:, 4] * orig_h * (qh / orig_h) / H
                all_labels.append(tile_lbl)

        merged_labels = (
            np.concatenate(all_labels, axis=0)
            if all_labels
            else np.zeros((0, 5), dtype=np.float32)
        )
        merged_labels = _clip_boxes(merged_labels)
        return canvas_pre, canvas_key, canvas_post, merged_labels

    # ------------------------------------------------------------------
    # Temporal MixUp
    # ------------------------------------------------------------------

    def _temporal_mixup(
        self,
        f_pre: np.ndarray,
        f_key: np.ndarray,
        f_post: np.ndarray,
        labels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Temporal MixUp: blend with a second random triplet using the SAME α
        for all three frames, preserving temporal structure.
        """
        assert self.triplet_provider is not None
        fp2, fk2, fpo2, lbl2 = self.triplet_provider()

        alpha = random.betavariate(8.0, 8.0)  # concentrated near 0.5

        H, W = self.img_size
        fp2 = cv2.resize(fp2, (W, H), interpolation=cv2.INTER_LINEAR)
        fk2 = cv2.resize(fk2, (W, H), interpolation=cv2.INTER_LINEAR)
        fpo2 = cv2.resize(fpo2, (W, H), interpolation=cv2.INTER_LINEAR)

        f_pre = cv2.resize(f_pre, (W, H), interpolation=cv2.INTER_LINEAR)
        f_key = cv2.resize(f_key, (W, H), interpolation=cv2.INTER_LINEAR)
        f_post = cv2.resize(f_post, (W, H), interpolation=cv2.INTER_LINEAR)

        mixed_pre = (
            alpha * f_pre.astype(np.float32) + (1 - alpha) * fp2.astype(np.float32)
        ).astype(np.uint8)
        mixed_key = (
            alpha * f_key.astype(np.float32) + (1 - alpha) * fk2.astype(np.float32)
        ).astype(np.uint8)
        mixed_post = (
            alpha * f_post.astype(np.float32) + (1 - alpha) * fpo2.astype(np.float32)
        ).astype(np.uint8)

        # Concatenate labels (both sets remain valid at reduced confidence)
        if len(lbl2):
            merged = np.concatenate([labels, lbl2], axis=0) if len(labels) else lbl2
        else:
            merged = labels

        return mixed_pre, mixed_key, mixed_post, merged


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _build_affine_matrix(
    scale: float,
    tx: float,
    ty: float,
    angle_deg: float,
    src_w: int,
    src_h: int,
    dst_w: int,
    dst_h: int,
) -> np.ndarray:
    """
    Build a 2×3 affine matrix that:
      1. Scales around the image centre
      2. Translates
      3. Rotates around the image centre

    Output is a float64 numpy array (2, 3).
    """
    cx, cy = src_w / 2.0, src_h / 2.0

    # Scale + rotation matrix around centre
    M_rot = cv2.getRotationMatrix2D((cx, cy), angle_deg, scale)

    # Add translation
    M_rot[0, 2] += tx + (dst_w - src_w) / 2.0
    M_rot[1, 2] += ty + (dst_h - src_h) / 2.0

    return M_rot
