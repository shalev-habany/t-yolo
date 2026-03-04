# Code Conventions

**Analysis Date:** 2026-03-04

---

## Code Style

**Language:** Python 3.12 (inferred from `__pycache__/*.cpython-312.pyc`)

**Formatter/Linter:** Ruff `0.15.4` (`.ruff_cache/` present). No `pyproject.toml` or `ruff.toml` config file found — Ruff is run with defaults.

**Line length:** Not explicitly configured; standard 88 (Ruff default) implied.

**Future imports:** Every module begins with `from __future__ import annotations` to enable postponed evaluation of type hints (PEP 563). This is mandatory in this codebase.

```python
# Required at top of every module
from __future__ import annotations
```

**Import order (observed pattern):**
1. `from __future__ import annotations`
2. Standard library (`argparse`, `math`, `os`, `pathlib`, `random`, `sys`, `time`, `typing`)
3. Third-party (`numpy`, `torch`, `cv2`, `yaml`, `ultralytics`)
4. Local project imports (`from models.x import X`, `from utils.x import Y`)

**`sys.path` manipulation:** Entry-point scripts (`train.py`, `val.py`, `smoke_test.py`) insert the project root at index 0 so that `models.*` and `utils.*` imports work without installation:
```python
sys.path.insert(0, str(Path(__file__).parent))
```

**Module-level `_HERE` / `_CONFIG_DIR` constants:** Used in model files for path resolution relative to the source file:
```python
_HERE = Path(__file__).parent
_CONFIG_DIR = _HERE.parent / "configs"
```

---

## Naming Conventions

**Classes:** `PascalCase`
- Public: `TYOLOv8`, `T2YOLOv8`, `TemporalDataset`, `TemporalAugmentor`, `FrameRegistrar`, `ECCRegistrar`
- Private (module-internal): leading underscore `_BackboneExtractor`, `_FPNNeck`, `_Registrar`, `_Augmentor`

**Functions:** `snake_case`
- Public: `build_model`, `build_dataset`, `build_optimizer`, `build_scheduler`, `temporal_collate_fn`, `build_registrar`, `evaluate`, `decode_predictions`
- Private helpers (module-internal): leading underscore — `_load_gray`, `_load_labels`, `_compute_tile_positions`, `_clip_labels_to_tile`, `_xywh2xyxy`, `_xyxy_iou`, `_gt_to_xyxy`, `_gt_area_px`, `_clip_boxes`, `_apply_affine_to_labels`, `_build_affine_matrix`, `_value_jitter_delta`, `_to_gray`
- Module-level constants: `SCREAMING_SNAKE_CASE` — `PASS`, `FAIL` in `smoke_test.py`; `_PRETRAIN_MAP`, `_MOTION_SCALE`, `_TINY_AREA_THRESHOLDS` (underscore-prefixed when module-private)

**Variables:** `snake_case`
- Tensor variables use abbreviations following ML conventions: `x_app`, `x_mot`, `f_pre`, `f_key`, `f_post`, `X_app`, `X_mot` (uppercase X for model inputs in batch dicts)
- Loop indices: single letters (`i`, `j`, `k`) or descriptive (`bi` for batch index, `thr_idx` for threshold index)
- Shape dimensions: `B`, `H`, `W`, `N`, `M`, `C` (uppercase single letters for tensor dimensions)

**Files:** `snake_case.py` — `t_yolov8.py`, `t2_yolov8.py`, `temporal_dataset.py`, `temporal_augmentation.py`, `frame_registration.py`

**Config/YAML files:** `snake_case.yaml` — `t_yolov8.yaml`, `t2_yolov8.yaml`, `visdrone.yaml`

---

## Patterns

### Module Structure Pattern
Every Python module follows this layout:
```
"""
module_name.py

One-paragraph description.

Paper reference (where applicable).

Usage:
    # Example usage
"""

from __future__ import annotations
# stdlib imports
# third-party imports
# local imports

# ---------------------------------------------------------------------------
# Section header (dashes, 75 chars)
# ---------------------------------------------------------------------------

# code
```

Section dividers use exactly this format:
```python
# ---------------------------------------------------------------------------
# Section Name
# ---------------------------------------------------------------------------
```

### Class Documentation Pattern
Classes use Google-style docstrings with an `Args:` section:
```python
class TemporalDataset(Dataset):
    """
    Brief summary.

    Args:
        split_root:  Description.
        temporal_shift:  Description.
    """
```

### Function Return Type Annotations
All public functions use full type annotations including return type:
```python
def build_model(
    model_type: str,
    scale: str,
    nc: int,
    app_weights: str | None,
    mot_weights: str | None,
    verbose: bool = True,
) -> TYOLOv8 | T2YOLOv8:
```

Private helpers on `val.py` use `Args:` / `Returns:` docstring format:
```python
def decode_predictions(
    raw: torch.Tensor,
    conf_thres: float = 0.001,
    ...
) -> list[torch.Tensor]:
    """
    Decode...

    Args:
        raw: (B, 4+nc, num_anchors) ...
    Returns:
        List of length B. Each element is ...
    """
```

### Factory Function Pattern
Stateless configuration-driven construction uses `build_*` factory functions (not class methods):
- `build_model()` in `train.py`
- `build_dataset()` in `train.py`
- `build_optimizer()` in `train.py`
- `build_scheduler()` in `train.py`
- `build_registrar()` in `utils/frame_registration.py`

### Protocol Pattern
Structural typing via `typing.Protocol` is used for dependency injection:
```python
# utils/temporal_dataset.py
class _Registrar(Protocol):
    def register(self, src: np.ndarray, ref: np.ndarray) -> np.ndarray: ...

class _Augmentor(Protocol):
    def __call__(self, f_pre, f_key, f_post, labels) -> tuple[...]: ...
```

### Union type syntax
Uses Python 3.10+ union syntax `X | Y` (enabled by `from __future__ import annotations`):
```python
def build_model(...) -> TYOLOv8 | T2YOLOv8:
model_type: str,
app_weights: str | None,
```

### Inline Comments
Inline comments are used heavily to annotate tensor shapes and algorithmic steps:
```python
preds = raw.permute(0, 2, 1).contiguous()  # (B, A, 4+nc)
batch["cls"] = labels[:, 1:2]  # (N, 1)
batch["bboxes"] = labels[:, 2:6]  # (N, 4)
```

### `__init__.py` Barrel Exports
Both `models/` and `utils/` use explicit `__all__` in `__init__.py`:
```python
# models/__init__.py
from models.t_yolov8 import TYOLOv8
from models.t2_yolov8 import T2YOLOv8
__all__ = ["TYOLOv8", "T2YOLOv8"]
```

### Lazy Imports
Some imports are deferred to inside functions to avoid circular imports or optional-dependency issues:
```python
# train.py build_dataset()
import cv2
from utils.temporal_dataset import _load_labels
```

---

## Error Handling

**Strategy:** Fail-fast with explicit, descriptive exceptions. No silent swallowing.

**FileNotFoundError for missing config/data:**
```python
if not _T_YAML.exists():
    raise FileNotFoundError(
        f"T-YOLOv8 config not found: {_T_YAML}\n"
        "Make sure configs/t_yolov8.yaml is present."
    )
```

**ValueError for invalid enum-style arguments:**
```python
raise ValueError(f"Unknown model type '{model_type}'. Use 't' or 't2'.")
raise ValueError(f"Unknown registration method '{method}'. Choose 'ecc' or 'sift'.")
```

**TypeError for wrong input types:**
```python
raise TypeError(
    "T2YOLOv8.forward() expects a dict or (X_app, X_mot) tuple. "
    f"Got {type(x)}."
)
```

**RuntimeError for data pipeline failures:**
```python
raise RuntimeError(
    f"No labelled frames found under {self.split_root}. "
    "Check that labels/ directories contain .txt files."
)
```

**Graceful degradation in frame registration** (not an exception — fall back silently):
```python
# utils/frame_registration.py
try:
    _, warp_matrix = cv2.findTransformECC(...)
except cv2.error:
    return src  # ECC failed, return unregistered
```

**FileNotFoundError for missing frames:**
```python
def _load_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load frame: {path}")
    return img
```

**`assert` statements** are used in private augmentation methods for contract checking (not in public APIs):
```python
assert self.triplet_provider is not None
```

**`type: ignore` comments** are used sparingly only where Protocol-typed attributes are assigned concrete instances:
```python
self.registrar = build_registrar(registration_method)  # type: ignore[assignment]
```

---

## Comments & Documentation

**Paper citations:** All modules, classes, and non-trivial algorithms include a reference to the paper section:
```python
# Paper §3.3
# Paper §4.1 — SGD settings
# paper Section 3.1
```

**Algorithm explanations in comments:** Complex tensor operations are annotated with their effect:
```python
# Per-class NMS: offset boxes by class id so boxes of different classes
# never suppress each other (standard torchvision approach).
max_wh = 4096.0
```

**Section headers in long functions:** Long functions use `# ---` comments to denote logical phases:
```python
# --- Device ---
# --- Load dataset config ---
# --- Model ---
# --- Datasets & loaders ---
# --- Optimiser + scheduler ---
# --- Training loop ---
```

---

*Convention analysis: 2026-03-04*
