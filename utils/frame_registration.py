"""
utils/frame_registration.py

Frame registration for moving-camera video sequences (e.g. VisDrone).
Aligns support frames f_{t-s} and f_{t+s} onto the reference frame f_t
using SIFT feature matching + RANSAC homography estimation.

Required because the motion stream (|f_{t-s} - f_t|) is only meaningful
when background pixels are spatially aligned. Without registration, every
pixel shows a difference due to camera motion, not object motion.
"""

import cv2
import numpy as np
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# SIFT-based homography estimator (accurate, ~5-15ms per pair on CPU)
# ---------------------------------------------------------------------------


class FrameRegistrar:
    """
    Registers a source frame onto a reference frame using SIFT + RANSAC.

    Usage:
        registrar = FrameRegistrar()
        aligned = registrar.register(src_frame, ref_frame)
    """

    def __init__(
        self,
        n_features: int = 2000,
        ratio_thresh: float = 0.75,
        min_matches: int = 10,
    ):
        self.sift = cv2.SIFT_create(nfeatures=n_features)  # type: ignore[attr-defined]
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.ratio_thresh = ratio_thresh
        self.min_matches = min_matches

    def register(
        self,
        src: np.ndarray,
        ref: np.ndarray,
    ) -> np.ndarray:
        """
        Warp `src` to align with `ref`.

        Args:
            src: Source frame to be warped. Shape (H, W) or (H, W, C). uint8.
            ref: Reference frame. Same shape as src.

        Returns:
            Warped source frame aligned to ref's coordinate system.
            If registration fails (too few matches), returns src unchanged.
        """
        h, w = ref.shape[:2]

        # Work in grayscale for feature detection
        src_gray = _to_gray(src)
        ref_gray = _to_gray(ref)

        kp1, des1 = self.sift.detectAndCompute(src_gray, None)
        kp2, des2 = self.sift.detectAndCompute(ref_gray, None)

        if des1 is None or des2 is None or len(kp1) < self.min_matches:
            return src  # fall back to unregistered

        # Lowe's ratio test
        matches = self.matcher.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < self.ratio_thresh * n.distance]

        if len(good) < self.min_matches:
            return src

        pts_src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts_ref = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(pts_src, pts_ref, cv2.RANSAC, 5.0)

        if H is None:
            return src

        return cv2.warpPerspective(src, H, (w, h))


# ---------------------------------------------------------------------------
# Fast ECC-based affine registration (faster, good for small camera motions)
# ---------------------------------------------------------------------------


class ECCRegistrar:
    """
    Registers frames using Enhanced Correlation Coefficient (ECC) minimization.
    Faster than SIFT for small, smooth camera motions (translation + rotation).
    Good fit for drone footage with stabilized cameras.

    Uses MOTION_EUCLIDEAN (rotation + translation, 3 DoF) by default.
    """

    def __init__(
        self,
        motion_type: int = cv2.MOTION_EUCLIDEAN,
        n_iterations: int = 50,
        termination_eps: float = 1e-4,
    ):
        self.motion_type = motion_type
        self.criteria: tuple = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            n_iterations,
            termination_eps,
        )

        if motion_type == cv2.MOTION_HOMOGRAPHY:
            self._warp_init = np.eye(3, 3, dtype=np.float32)
        else:
            self._warp_init = np.eye(2, 3, dtype=np.float32)

    def register(
        self,
        src: np.ndarray,
        ref: np.ndarray,
    ) -> np.ndarray:
        h, w = ref.shape[:2]
        src_gray = _to_gray(src).astype(np.float32)
        ref_gray = _to_gray(ref).astype(np.float32)

        warp_matrix = self._warp_init.copy()

        try:
            _, warp_matrix = cv2.findTransformECC(
                ref_gray,
                src_gray,
                warp_matrix,
                self.motion_type,
                self.criteria,
            )
        except cv2.error:
            return src  # ECC failed (insufficient texture), return unregistered

        if self.motion_type == cv2.MOTION_HOMOGRAPHY:
            aligned = cv2.warpPerspective(
                src, warp_matrix, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
            )
        else:
            aligned = cv2.warpAffine(
                src, warp_matrix, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
            )
        return aligned


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.ndim == 3 and img.shape[2] == 1:
        return img[:, :, 0]
    return img


def build_registrar(method: str = "ecc") -> object:
    """
    Factory function.

    Args:
        method: 'ecc' (fast, good for drone) or 'sift' (robust, slower).

    Returns:
        A registrar object with a .register(src, ref) -> np.ndarray method.
    """
    if method == "sift":
        return FrameRegistrar()
    elif method == "ecc":
        return ECCRegistrar()
    else:
        raise ValueError(
            f"Unknown registration method '{method}'. Choose 'ecc' or 'sift'."
        )
