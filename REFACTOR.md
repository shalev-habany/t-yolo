# T-YOLO Refactoring Plan

## Goals

1. Replace the current collection of large monolithic scripts with small, single-responsibility modules and classes.
2. Eliminate nested functions (closures) that cannot be pickled by Python's `spawn` multiprocessing.
3. De-duplicate logic that appears in multiple files (device resolution, weight transfer).
4. Produce a clean `README.md` that lists every command a user needs to run.
5. Delete the redundant `scripts/convert_visdrone.py` subprocess wrapper.
6. Fix the `configs/visdrone.yaml` bug where `train:` points at `sequences/val`.

---

## Current structure (problems annotated)

```
T-YOLO/
├── train.py               521 lines — 7 responsibilities mixed together
│                          nested lr_lambda closure (spawn-unsafe pattern)
│                          device resolution duplicated with val.py / smoke_test.py
│                          weight-transfer boilerplate duplicated with models/
├── val.py                 512 lines — is both a library (imported by train.py)
│                          and a CLI; _compute_ap nested function (untestable)
│                          device resolution duplicated
├── smoke_test.py          device resolution duplicated
│
├── configs/
│   ├── t_yolov8.yaml      was missing data/epochs/batch_size keys (fixed previously)
│   ├── t2_yolov8.yaml     fine
│   └── visdrone.yaml      BUG: train: sequences/val  (should be sequences/train)
│
├── models/
│   ├── t_yolov8.py        load_pretrained has inline weight-transfer loop
│   └── t2_yolov8.py       _load_backbone_weights is near-duplicate of above
│
├── utils/
│   ├── temporal_dataset.py  510 lines — tiling helpers mixed with dataset logic
│   ├── temporal_augmentation.py  dead function _value_jitter (never called)
│   └── frame_registration.py  pointless if/else branch (both arms identical)
│
├── data/
│   └── visdrone_converter.py  fine — the real implementation
│
└── scripts/
    ├── download_visdrone.sh  fine
    └── convert_visdrone.py   REDUNDANT — just calls visdrone_converter.py via subprocess
```

---

## Target structure

```
T-YOLO/
├── README.md                    NEW  commands-only user guide
├── REFACTOR.md                  this file
├── train.py                     ~40 lines — thin CLI only
├── val.py                       ~50 lines — thin CLI only
├── smoke_test.py                minor: fix import paths for moved tiling helpers
│
├── configs/                     unchanged except one-line bug fix
│   ├── t_yolov8.yaml
│   ├── t2_yolov8.yaml
│   └── visdrone.yaml            FIX: train: sequences/val -> sequences/train
│
├── models/
│   ├── __init__.py              unchanged
│   ├── t_yolov8.py              load_pretrained delegates to utils/weights.py
│   └── t2_yolov8.py             _load_backbone_weights removed, delegates to utils/weights.py
│
├── core/
│   └── trainer.py               NEW  Trainer class — all training logic
│
├── utils/
│   ├── __init__.py              updated exports
│   ├── device.py                NEW  resolve_device() — shared by train/val/smoke_test
│   ├── weights.py               NEW  transfer_weights() — shared by both model files
│   ├── metrics.py               NEW  extracted from val.py: evaluate(), decode_predictions(),
│   │                                 match_predictions(), _compute_ap(), AP helpers
│   ├── tiling.py                NEW  extracted from temporal_dataset.py:
│   │                                 _compute_tile_positions(), _clip_labels_to_tile()
│   ├── temporal_dataset.py      slimmed: imports tiling from utils/tiling.py
│   ├── temporal_augmentation.py  remove dead _value_jitter function
│   └── frame_registration.py    fix pointless if/else in FrameRegistrar.register
│
├── data/
│   └── visdrone_converter.py    unchanged
│
└── scripts/
    ├── download_visdrone.sh     unchanged
    └── convert_visdrone.py      DELETED
```

---

## Changes, file by file

---

### 1. `README.md` — NEW

Commands-only guide. Sections: Install, Download data, Convert data, Train, Validate.
No lengthy prose. Each section is a single code block.

**Content outline:**

```
## Install
pip install -r requirements.txt

## Download VisDrone
bash scripts/download_visdrone.sh --dst data/raw_visdrone

## Convert VisDrone
# Full dataset root (all splits present)
python data/visdrone_converter.py --src data/raw_visdrone --dst data/VisDrone

# Single split only (e.g. val only)
python data/visdrone_converter.py --src data/VisDrone2019-VID-val --dst data/VisDrone

## Train
# Single-stream T-YOLOv8 (nano)
python train.py --config configs/t_yolov8.yaml --scale n

# Two-stream T2-YOLOv8 (nano)
python train.py --config configs/t2_yolov8.yaml --scale n

# Override dataset path without editing YAML
python train.py --config configs/t_yolov8.yaml --data configs/visdrone.yaml --scale n

# T2: pass a pretrained T-YOLOv8 checkpoint for the appearance stream
python train.py --config configs/t2_yolov8.yaml --scale n \
    --app-weights runs/t_yolov8n_visdrone/weights/best.pt

## Validate
python val.py --weights runs/t_yolov8n_visdrone/weights/best.pt \
              --config configs/t_yolov8.yaml

## Smoke test (no real data required)
python smoke_test.py --device cpu
```

---

### 2. `core/trainer.py` — NEW

Extract everything except CLI argument parsing out of `train.py` into a `Trainer` class.

**Full public API:**

```python
class Trainer:
    def __init__(self, cfg: dict, app_weights: str | None, mot_weights: str | None) -> None
    def run(self) -> None
```

**Internal organisation (all become private methods or module-level helpers):**

| Current location | New location |
|---|---|
| `set_seed()` in `train.py` | `Trainer._set_seed()` static method |
| `build_model()` in `train.py` | `Trainer._build_model()` |
| `_TripletProvider` class in `train.py` | moved to `core/trainer.py` (same file) |
| `build_dataset()` in `train.py` | `Trainer._build_dataset()` |
| `build_optimizer()` in `train.py` | `Trainer._build_optimizer()` |
| `build_scheduler()` + nested `lr_lambda` in `train.py` | `Trainer._build_scheduler()` + module-level `_cosine_lr(epoch, epochs, lrf)` replacing the closure |
| `warmup_lr()` in `train.py` | `Trainer._warmup_lr()` static method |
| `train()` loop in `train.py` | `Trainer.run()` |

**Why replace `lr_lambda` closure with a module-level function:**

The existing `build_scheduler` contains:

```python
def build_scheduler(optimizer, cfg, epochs):
    lrf = cfg.get("lrf", 0.01)

    def lr_lambda(epoch: int) -> float:          # <-- closure over lrf and epochs
        return max(lrf, 0.5 * (1.0 + math.cos(math.pi * epoch / epochs)))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)
```

`LambdaLR` pickles the `lr_lambda` function when saving checkpoints. Closures are not
reliably picklable across Python versions. Replace with:

```python
# module-level — always picklable
def _cosine_lr(epoch: int, epochs: int, lrf: float) -> float:
    return max(lrf, 0.5 * (1.0 + math.cos(math.pi * epoch / epochs)))

# inside Trainer._build_scheduler:
fn = functools.partial(_cosine_lr, epochs=epochs, lrf=lrf)
return LambdaLR(optimizer, lr_lambda=fn)
```

`functools.partial` of a module-level function is always picklable.

---

### 3. `train.py` — becomes thin CLI (~40 lines)

After extraction, `train.py` only contains:

```python
import argparse, yaml, sys
from pathlib import Path
from core.trainer import Trainer

def parse_args() -> argparse.Namespace: ...   # unchanged from current

if __name__ == "__main__":
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    # apply CLI overrides (data, model, scale, epochs, batch_size, device)
    ...
    Trainer(cfg, app_weights=args.app_weights, mot_weights=args.mot_weights).run()
```

---

### 4. `utils/metrics.py` — NEW

Move everything evaluation-related out of `val.py`. This makes the functions independently
importable and testable without pulling in the CLI scaffolding.

**Functions moved here:**

| Function | Notes |
|---|---|
| `decode_predictions()` | unchanged |
| `_xywh2xyxy()` | unchanged |
| `_xyxy_iou()` | unchanged |
| `match_predictions()` | unchanged |
| `_gt_to_xyxy()` | unchanged |
| `_gt_area_px()` | unchanged |
| `_TINY_AREA_THRESHOLDS` | unchanged |
| `evaluate()` | `_compute_ap` nested function promoted (see below) |

**Promoting `_compute_ap` out of `evaluate()`:**

Currently `_compute_ap` is a nested function inside `evaluate()` that closes over
`iou_eval_thresholds`. It does real work (concatenation + `ap_per_class`) and is called
5 times. Nested functions cannot be independently tested.

Replace with:

```python
# module-level — no closure, receives everything it needs as arguments
def _compute_ap(
    tp_list: list,
    conf_list: list,
    pred_cls_list: list,
    gt_cls_list: list,
    iou_thresholds: np.ndarray,
) -> tuple[float, float, float, float, float]:
    ...
```

`evaluate()` passes `iou_eval_thresholds` explicitly at each call site instead of relying
on closure capture.

---

### 5. `val.py` — becomes thin CLI (~50 lines)

After extraction, `val.py` only contains:

```python
import argparse, yaml, sys, torch
from pathlib import Path
from models.t_yolov8 import TYOLOv8
from models.t2_yolov8 import T2YOLOv8
from utils.metrics import evaluate
from utils.temporal_dataset import TemporalDataset, temporal_collate_fn
from utils.device import resolve_device

def parse_args() -> argparse.Namespace: ...   # unchanged from current

def main() -> None: ...                       # unchanged logic, uses resolve_device()

if __name__ == "__main__":
    main()
```

`train.py` import changes from `from val import evaluate` to `from utils.metrics import evaluate`.

---

### 6. `utils/device.py` — NEW

The device-resolution block appears identically in three files:

**In `train.py`:**
```python
device_str = str(cfg.get("device", "0"))
if device_str == "cpu":
    device = torch.device("cpu")
elif torch.cuda.is_available():
    device = torch.device(f"cuda:{device_str}")
else:
    print("CUDA not available, falling back to CPU")
    device = torch.device("cpu")
```

**In `val.py`:**
```python
device_str = str(cfg.get("device", "0"))
device = (
    torch.device(f"cuda:{device_str}")
    if device_str != "cpu" and torch.cuda.is_available()
    else torch.device("cpu")
)
```

**In `smoke_test.py`:**
Same pattern, third time.

**New shared function:**

```python
# utils/device.py
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
```

All three callers replace their block with `device = resolve_device(device_str)`.

---

### 7. `utils/weights.py` — NEW

The weight-transfer pattern appears twice with near-identical logic:

**In `models/t_yolov8.py` (`load_pretrained`):**
```python
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
```

**In `models/t2_yolov8.py` (`_load_backbone_weights`):**
Same loop, plus a key remapping step: `model.{i}.xxx` → `layers.{i}.xxx`.

**New shared function:**

```python
# utils/weights.py
import torch.nn as nn
from ultralytics.utils import LOGGER

def transfer_weights(
    src_state: dict,
    dst_model: nn.Module,
    key_map: callable | None = None,
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
    transfer = {}
    skipped = []

    for k, v in src_state.items():
        dst_key = key_map(k) if key_map is not None else k
        if dst_key in dst_state and dst_state[dst_key].shape == v.shape:
            transfer[dst_key] = v
        else:
            skipped.append(k)

    dst_state.update(transfer)
    dst_model.load_state_dict(dst_state)

    if skipped:
        LOGGER.info(f"transfer_weights: skipped {len(skipped)} tensors (shape mismatch or key not found)")
    LOGGER.info(f"transfer_weights: transferred {len(transfer)}/{len(src_state)} tensors")
    return len(transfer), len(src_state)
```

**`models/t_yolov8.py` after:**
```python
from utils.weights import transfer_weights

def load_pretrained(self, weights_path=None):
    ...
    pretrained, _ = load_checkpoint(weights_path, device="cpu")
    transfer_weights(pretrained.state_dict(), self)
    return self
```

**`models/t2_yolov8.py` after:**
```python
from utils.weights import transfer_weights

# key_map for backbone: "model.0.conv.weight" -> "layers.0.conv.weight"
def _backbone_key_map(k: str) -> str:
    if k.startswith("model."):
        return "layers." + k[len("model."):]
    return k

# inside T2YOLOv8.load_pretrained:
transfer_weights(coco_state, self.app_backbone, key_map=_backbone_key_map)
transfer_weights(coco_state, self.mot_backbone, key_map=_backbone_key_map)
```

The module-level `_load_backbone_weights` function in `t2_yolov8.py` is deleted entirely.

---

### 8. `utils/tiling.py` — NEW

`temporal_dataset.py` is 510 lines. The tiling helpers are self-contained and can be
extracted without changing any logic:

**Functions moved here:**

| Function | Current lines in temporal_dataset.py |
|---|---|
| `_compute_tile_positions(h, w, tile_h, tile_w, overlap)` | ~30 lines |
| `_clip_labels_to_tile(labels, y0, x0, y1, x1, min_visibility)` | ~25 lines |

`temporal_dataset.py` replaces its definitions with:
```python
from utils.tiling import _compute_tile_positions, _clip_labels_to_tile
```

`smoke_test.py` updates its import from `utils.temporal_dataset` to `utils.tiling`.

---

### 9. `utils/temporal_augmentation.py` — remove dead code

`_value_jitter` (lines ~54–61) is defined but never called. Only `_value_jitter_delta`
is used directly by `TemporalAugmentor.__call__`. Delete `_value_jitter`.

```python
# DELETE this function entirely:
def _value_jitter(frame: np.ndarray, max_delta: int = 30) -> np.ndarray:
    delta = random.randint(-max_delta, max_delta)
    return _value_jitter_delta(frame, delta)
```

---

### 10. `utils/frame_registration.py` — fix dead branch

In `FrameRegistrar.register`, both arms of an `if src.ndim == 2` are identical:

```python
# CURRENT (both branches do the same thing — the if is pointless):
if src.ndim == 2:
    return cv2.warpPerspective(src, H, (w, h))
else:
    return cv2.warpPerspective(src, H, (w, h))
```

Fix: collapse to a single unconditional call.

```python
# AFTER:
return cv2.warpPerspective(src, H, (w, h))
```

---

### 11. `configs/visdrone.yaml` — fix train split bug

```yaml
# BEFORE (wrong — trains on validation data):
train: sequences/val

# AFTER:
train: sequences/train
```

---

### 12. `scripts/convert_visdrone.py` — DELETE

This file only exists to call `data/visdrone_converter.py` via `subprocess.run`. It adds
an unnecessary process boundary, duplicates the argument definitions, and will silently
go out of sync if the converter's interface changes.

`data/visdrone_converter.py` already has a `if __name__ == "__main__"` block and is
directly runnable. Users call it directly:

```bash
python data/visdrone_converter.py --src data/raw_visdrone --dst data/VisDrone
```

---

### 13. `utils/__init__.py` — updated exports

Add the new modules to the package's public API:

```python
from utils.device import resolve_device
from utils.weights import transfer_weights
from utils.metrics import evaluate, decode_predictions
from utils.tiling import _compute_tile_positions, _clip_labels_to_tile
from utils.frame_registration import FrameRegistrar, ECCRegistrar, build_registrar
from utils.temporal_dataset import TemporalDataset, temporal_collate_fn
from utils.temporal_augmentation import TemporalAugmentor
```

---

## Migration checklist

The items below are ordered so that each step compiles and is independently testable
before the next step begins. Steps that touch the same file are grouped.

```
[ ] 1.  Create utils/device.py  (resolve_device)
[ ] 2.  Replace device block in train.py with resolve_device()
[ ] 3.  Replace device block in val.py with resolve_device()
[ ] 4.  Replace device block in smoke_test.py with resolve_device()

[ ] 5.  Create utils/weights.py  (transfer_weights)
[ ] 6.  Update models/t_yolov8.py  — load_pretrained uses transfer_weights()
[ ] 7.  Update models/t2_yolov8.py — remove _load_backbone_weights, use transfer_weights()

[ ] 8.  Create utils/tiling.py  (move _compute_tile_positions, _clip_labels_to_tile)
[ ] 9.  Update utils/temporal_dataset.py — import from utils/tiling.py
[ ] 10. Update smoke_test.py — fix import path for tiling helpers

[ ] 11. Create utils/metrics.py  (move all evaluation logic from val.py, promote _compute_ap)
[ ] 12. Slim val.py to thin CLI, import evaluate from utils/metrics
[ ] 13. Update train.py — change `from val import evaluate` to `from utils.metrics import evaluate`

[ ] 14. Create core/__init__.py  (empty)
[ ] 15. Create core/trainer.py  (Trainer class, _TripletProvider, _cosine_lr)
[ ] 16. Slim train.py to thin CLI, import Trainer from core/trainer

[ ] 17. Fix utils/temporal_augmentation.py — delete _value_jitter
[ ] 18. Fix utils/frame_registration.py — collapse pointless if/else in FrameRegistrar.register
[ ] 19. Fix configs/visdrone.yaml — train: sequences/val -> sequences/train

[ ] 20. Delete scripts/convert_visdrone.py

[ ] 21. Update utils/__init__.py — add new module exports

[ ] 22. Write README.md

[ ] 23. Run smoke_test.py --device cpu  (full regression)
```

---

## Line-count comparison

| File | Before | After |
|---|---|---|
| `train.py` | 521 | ~40 |
| `val.py` | 512 | ~50 |
| `core/trainer.py` | — | ~280 |
| `utils/metrics.py` | — | ~220 |
| `utils/device.py` | — | ~20 |
| `utils/weights.py` | — | ~40 |
| `utils/tiling.py` | — | ~65 |
| `utils/temporal_dataset.py` | 510 | ~420 |
| `models/t_yolov8.py` | 170 | ~130 |
| `models/t2_yolov8.py` | ~540 | ~490 |
| `scripts/convert_visdrone.py` | 92 | deleted |
| `utils/temporal_augmentation.py` | ~210 | ~200 |
| `utils/frame_registration.py` | ~100 | ~95 |

Total lines: **~2655 before → ~2050 after** (roughly −23%, with better separation of concerns).

---

## What is NOT changing

- All model architecture code (backbone, neck, head, forward pass logic)
- `data/visdrone_converter.py`
- `scripts/download_visdrone.sh`
- All YAML config keys and their semantics (only one value changes: the train split path)
- The public CLI interface of `train.py` and `val.py` (same flags, same behaviour)
- `smoke_test.py` logic (only import paths change)
