# Testing

**Analysis Date:** 2026-03-04

---

## Framework

**Runner:** No formal test framework (pytest, unittest) is installed or configured. Testing is done via a **custom smoke test runner** in `smoke_test.py`.

**Assertion style:** Python built-in `assert` statements with descriptive f-string messages.

**Run commands:**
```bash
python smoke_test.py              # Run all smoke tests (CPU)
python smoke_test.py --device 0  # Run on GPU device 0
python smoke_test.py --device cpu # Explicitly force CPU
```

**Exit codes:** The runner returns `sys.exit(1)` on any failure, making it CI-compatible.

**No test discovery:** There are no `tests/` directories, `*_test.py` files, or `test_*.py` files. All tests live in `smoke_test.py` at the project root.

---

## Structure

**Single test file:** `smoke_test.py` (266 lines) — contains all test functions plus runner harness.

**Test function naming:** `test_<subject>` — all test functions are prefixed with `test_`:
```
test_t_yolov8_forward(device)
test_t2_yolov8_forward(scale, device)
test_tiling_helpers()
test_temporal_augmentor()
test_collate_fn()
test_decode_predictions()
```

**Runner harness pattern:**
```python
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
```

**Test registration pattern:** Tests are collected as `(name, callable)` tuples in `main()` and executed sequentially:
```python
tests = [
    ("TYOLOv8(n) forward+loss+backward+inference", lambda: test_t_yolov8_forward(device)),
    ("T2YOLOv8(n) forward+loss+backward+inference", lambda: test_t2_yolov8_forward("n", device)),
    ("T2YOLOv8(x) forward+loss+backward", lambda: test_t2_yolov8_forward("x", device)),
    ("Tiling helpers (_compute_tile_positions, _clip_labels_to_tile)", test_tiling_helpers),
    ("TemporalAugmentor (spatial augmentations)", test_temporal_augmentor),
    ("temporal_collate_fn batching", test_collate_fn),
    ("decode_predictions (NMS)", test_decode_predictions),
]
```

**Test scope — 7 smoke tests total:**
| Test | What is verified |
|------|-----------------|
| `test_t_yolov8_forward` | `TYOLOv8(n)` train loss shape `(3,)`, backward pass, eval output shape `(1,14,2100)` |
| `test_t2_yolov8_forward` (n+x) | `T2YOLOv8` train loss shape, backward, eval output shape `(1,14,A)` |
| `test_tiling_helpers` | `_compute_tile_positions` tile count/size, `_clip_labels_to_tile` keep/drop logic |
| `test_temporal_augmentor` | `TemporalAugmentor` output shapes `(320,320)`, label column count `5` |
| `test_collate_fn` | `temporal_collate_fn` stacked shapes, batch index prepend |
| `test_decode_predictions` | `decode_predictions` returns list of length `B`, each element `(N,6)` |

---

## Mocking

**No mock library used.** The codebase avoids mocking entirely. Instead:

**Real tensor inputs** with known-safe values are constructed in-test:
```python
# TYOLOv8 forward test — synthetic zero-image batch
m = TYOLOv8(scale="n", nc=10, verbose=False).to(device)
x = torch.zeros(1, 3, 320, 320, device=device)
batch = {
    "img": x,
    "cls": torch.zeros(2, 1, device=device),
    "bboxes": torch.tensor(
        [[0.5, 0.5, 0.1, 0.1], [0.3, 0.3, 0.05, 0.05]], device=device
    ),
    "batch_idx": torch.tensor([0.0, 0.0], device=device),
}
loss, items = m.loss(batch)
```

**Controlled probabilities** force deterministic augmentation outcomes (instead of mocking `random`):
```python
aug = TemporalAugmentor(
    img_size=(320, 320),
    hflip_p=1.0,   # force flip ON
    vflip_p=1.0,
    scale_range=(1.0, 1.0),  # no scaling
    translate_frac=0.0,      # no translation
    mosaic_p=0.0,            # disable
    mixup_p=0.0,             # disable
)
```

**Near-zero logits** are used to produce known-sparse NMS outputs in `test_decode_predictions`:
```python
raw = torch.randn(1, 14, 2100)
raw[:, 4:, :] = -10.0   # suppress all classes
raw[0, 4, 0] = 5.0      # activate exactly one anchor
preds = decode_predictions(raw, conf_thres=0.01, ...)
```

**No filesystem mocking:** Tests that exercise I/O-heavy paths (`TemporalDataset`, `FrameRegistrar`) are not tested — they require real data on disk and are out of scope for the smoke test.

---

## Coverage

**Requirements:** None enforced. No coverage configuration, no CI pipeline, no threshold.

**View coverage (manual):**
```bash
# Install coverage if needed
pip install pytest pytest-cov

# Run with coverage (requires adapting smoke_test.py to pytest, or use coverage run)
coverage run smoke_test.py
coverage report
```

**What IS tested:**
- Model forward pass (training + inference mode) for `TYOLOv8` and `T2YOLOv8` at scales `n` and `x`
- Loss computation shape and backward pass (gradient flow)
- Tiling helper functions (`_compute_tile_positions`, `_clip_labels_to_tile`)
- `TemporalAugmentor` spatial augmentation output shapes
- `temporal_collate_fn` batching and label index prepend
- `decode_predictions` NMS output shape

**What is NOT tested:**
- `TemporalDataset.__getitem__` (requires real disk data)
- `FrameRegistrar.register` / `ECCRegistrar.register` (requires real frames)
- `evaluate()` in `val.py` (requires a full dataset + model checkpoint)
- `train()` loop in `train.py` (no integration test)
- `load_pretrained()` weight transfer (requires network or local checkpoint)
- `_load_backbone_weights()` key mapping
- Error paths (e.g. missing config file, invalid model type)
- Temporal Mosaic / MixUp augmentations (require a `triplet_provider`)

---

## Test Design Patterns

**Shape assertions** are the dominant correctness check:
```python
assert loss.shape == (3,), f"Expected loss shape (3,), got {loss.shape}"
assert out.shape == (1, 14, 2100), f"Expected (1,14,2100), got {out.shape}"
assert out["X_app"].shape == (2, 3, 320, 320)
assert out["labels"].shape == (1, 6)
```

**Count assertions** verify filtering logic:
```python
assert len(tiles) > 0, "No tiles generated"
assert len(tiles_small) == 1, f"Expected 1 tile for small image, got {len(tiles_small)}"
assert len(clipped) == 1, "Box in tile should be kept"
assert len(clipped_out) == 0, "Box outside tile should be dropped"
```

**Value assertions** check specific tensor values:
```python
assert out["labels"][0, 0] == 0.0  # batch index 0
```

**Backward-pass guard** (no assertion, just call — exception = failure):
```python
loss.sum().backward()  # any gradient error surfaces as test failure
```

**Device parametrization** via CLI argument, passed into each test as `device: torch.device`:
```python
python smoke_test.py --device cpu
python smoke_test.py --device 0    # CUDA device 0
```

---

*Testing analysis: 2026-03-04*
