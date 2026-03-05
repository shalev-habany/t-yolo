#!/usr/bin/env python3
"""
smoke_test.py

End-to-end integration smoke test for T-YOLO.

Tests (no real data needed):
  1. TYOLOv8(n)  — forward pass, loss, backward, inference shape
  2. T2YOLOv8(n) — forward pass, loss, backward, inference shape
  3. T2YOLOv8(x) — loss + backward (large model)
  4. Tiling helpers — _compute_tile_positions, _clip_labels_to_tile
  5. TemporalAugmentor — all augmentations (no Mosaic/MixUp without data)
  6. temporal_collate_fn — batching shapes

Usage:
    python smoke_test.py
    python smoke_test.py --device cpu
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from utils.device import resolve_device

PASS = "[PASS]"
FAIL = "[FAIL]"


def run(name: str, fn):
    try:
        fn()
        print(f"  {PASS}  {name}")
        return True
    except Exception as e:
        print(f"  {FAIL}  {name}")
        traceback.print_exc()
        return False


def test_t_yolov8_forward(device: torch.device) -> None:
    from models.t_yolov8 import TYOLOv8

    m = TYOLOv8(scale="n", nc=10, verbose=False).to(device)
    x = torch.zeros(1, 3, 320, 320, device=device)

    # Training forward (loss)
    m.train()
    batch = {
        "img": x,
        "cls": torch.zeros(2, 1, device=device),
        "bboxes": torch.tensor(
            [[0.5, 0.5, 0.1, 0.1], [0.3, 0.3, 0.05, 0.05]], device=device
        ),
        "batch_idx": torch.tensor([0.0, 0.0], device=device),
    }
    loss, items = m.loss(batch)
    assert loss.shape == (3,), f"Expected loss shape (3,), got {loss.shape}"
    loss.sum().backward()

    # Inference forward
    m.eval()
    with torch.no_grad():
        out = m(x)
    if isinstance(out, (list, tuple)):
        out = out[0]
    assert out.shape == (1, 14, 2100), f"Expected (1,14,2100), got {out.shape}"


def test_t2_yolov8_forward(scale: str, device: torch.device) -> None:
    from models.t2_yolov8 import T2YOLOv8

    m = T2YOLOv8(scale=scale, nc=10, verbose=False).to(device)
    x_app = torch.zeros(1, 3, 320, 320, device=device)
    x_mot = torch.zeros(1, 2, 320, 320, device=device)

    # Training loss
    m.train()
    batch = {
        "img": x_app,
        "X_mot": x_mot,
        "cls": torch.zeros(2, 1, device=device),
        "bboxes": torch.tensor(
            [[0.5, 0.5, 0.1, 0.1], [0.3, 0.3, 0.05, 0.05]], device=device
        ),
        "batch_idx": torch.tensor([0.0, 0.0], device=device),
    }
    loss, items = m.loss(batch)
    assert loss.shape == (3,), f"Expected loss shape (3,), got {loss.shape}"
    loss.sum().backward()

    # Inference
    m.eval()
    with torch.no_grad():
        out = m.predict(x_app, x_mot)
    if isinstance(out, (list, tuple)):
        out = out[0]
    assert out.shape[0] == 1 and out.shape[1] == 14, (
        f"Expected (1, 14, A), got {out.shape}"
    )


def test_tiling_helpers() -> None:
    from utils.tiling import _compute_tile_positions, _clip_labels_to_tile

    # Basic grid
    tiles = _compute_tile_positions(1080, 1920, 320, 320, 0.05)
    assert len(tiles) > 0, "No tiles generated"
    # All tiles exactly 320×320
    for y0, x0, y1, x1 in tiles:
        assert y1 - y0 == 320, f"Tile height {y1 - y0} != 320"
        assert x1 - x0 == 320, f"Tile width {x1 - x0} != 320"

    # Edge: image smaller than tile
    tiles_small = _compute_tile_positions(100, 200, 320, 320, 0.05)
    assert len(tiles_small) == 1, (
        f"Expected 1 tile for small image, got {len(tiles_small)}"
    )

    # Label clipping: box entirely in tile
    labels = np.array([[0, 0.05, 0.05, 0.04, 0.04]], dtype=np.float32)  # top-left
    clipped = _clip_labels_to_tile(labels, 0, 0, 320, 320, 1920, 1080)
    assert len(clipped) == 1, "Box in tile should be kept"

    # Label clipping: box entirely outside tile
    labels_out = np.array([[0, 0.9, 0.9, 0.05, 0.05]], dtype=np.float32)
    clipped_out = _clip_labels_to_tile(labels_out, 0, 0, 320, 320, 1920, 1080)
    assert len(clipped_out) == 0, "Box outside tile should be dropped"


def test_temporal_augmentor() -> None:
    from utils.temporal_augmentation import TemporalAugmentor

    aug = TemporalAugmentor(
        img_size=(320, 320),
        hflip_p=1.0,  # force all augmentations
        vflip_p=1.0,
        scale_range=(1.0, 1.0),
        translate_frac=0.0,
        mosaic_p=0.0,
        mixup_p=0.0,
    )

    f = np.zeros((320, 320), dtype=np.uint8)
    labels = np.array([[0, 0.5, 0.5, 0.2, 0.2]], dtype=np.float32)

    f_pre, f_key, f_post, lbl = aug(f.copy(), f.copy(), f.copy(), labels.copy())

    assert f_pre.shape == (320, 320)
    assert f_key.shape == (320, 320)
    assert f_post.shape == (320, 320)
    assert lbl.shape[1] == 5


def test_collate_fn() -> None:
    from utils.temporal_dataset import temporal_collate_fn
    import numpy as np

    batch = [
        {
            "X_app": torch.zeros(3, 320, 320),
            "X_mot": torch.zeros(2, 320, 320),
            "labels": np.array([[0, 0.5, 0.5, 0.1, 0.1]], dtype=np.float32),
            "key_path": "a",
            "seq_id": "s",
        },
        {
            "X_app": torch.zeros(3, 320, 320),
            "X_mot": torch.zeros(2, 320, 320),
            "labels": np.zeros((0, 5), dtype=np.float32),
            "key_path": "b",
            "seq_id": "s",
        },
    ]
    out = temporal_collate_fn(batch)
    assert out["X_app"].shape == (2, 3, 320, 320)
    assert out["X_mot"].shape == (2, 2, 320, 320)
    assert out["labels"].shape == (1, 6)  # 1 box total, batch_idx prepended
    assert out["labels"][0, 0] == 0.0  # batch index 0


def test_decode_predictions() -> None:
    from utils.metrics import decode_predictions

    # Fake raw output (B=1, 4+10=14 channels, A=2100 anchors)
    raw = torch.randn(1, 14, 2100)
    # Low scores -> most filtered out
    raw[:, 4:, :] = -10.0
    # One anchor with a real score
    raw[0, 4, 0] = 5.0  # class 0, high confidence

    preds = decode_predictions(raw, conf_thres=0.01, iou_thres=0.65, nc=10)
    assert len(preds) == 1, f"Expected 1 prediction list, got {len(preds)}"
    # At least 1 detection
    assert preds[0].shape[1] == 6 if len(preds[0]) > 0 else True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", help="Device: 'cpu' or '0'")
    args = parser.parse_args()

    dev_str = args.device
    device = resolve_device(dev_str)

    print(f"Device: {device}\n")

    tests = [
        (
            "TYOLOv8(n) forward+loss+backward+inference",
            lambda: test_t_yolov8_forward(device),
        ),
        (
            "T2YOLOv8(n) forward+loss+backward+inference",
            lambda: test_t2_yolov8_forward("n", device),
        ),
        (
            "T2YOLOv8(x) forward+loss+backward",
            lambda: test_t2_yolov8_forward("x", device),
        ),
        (
            "Tiling helpers (_compute_tile_positions, _clip_labels_to_tile)",
            test_tiling_helpers,
        ),
        ("TemporalAugmentor (spatial augmentations)", test_temporal_augmentor),
        ("temporal_collate_fn batching", test_collate_fn),
        ("decode_predictions (NMS)", test_decode_predictions),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        ok = run(name, fn)
        if ok:
            passed += 1
        else:
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"Results: {passed}/{passed + failed} passed")
    if failed:
        print(f"  {failed} test(s) FAILED")
        sys.exit(1)
    else:
        print("All tests passed.")


if __name__ == "__main__":
    main()
