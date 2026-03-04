# Architecture

**Analysis Date:** 2026-03-04

## Pattern

**Overall:** Research ML pipeline — flat module layout with clear separation between model definitions, data utilities, and CLI entry points. No application-layer abstraction; the research paper's design maps directly to code.

**Key Characteristics:**
- Two model variants built on top of ultralytics YOLOv8: single-stream (`TYOLOv8`) and two-stream (`T2YOLOv8`)
- Temporal context is the central architectural concern — all data loading, augmentation, and model design are organized around processing frame triplets `(f_{t-s}, f_t, f_{t+s})`
- Training and evaluation are orchestrated by standalone scripts (`train.py`, `val.py`), not a framework trainer
- Configuration is YAML-driven; CLI flags override config values at runtime

---

## Layers

**Model Layer:**
- Purpose: Define neural network architectures for temporal tiny-object detection
- Location: `models/`
- Contains: `TYOLOv8` (single-stream), `T2YOLOv8` (two-stream), internal submodules `_BackboneExtractor`, `_FPNNeck`
- Depends on: `ultralytics.nn` (DetectionModel, C2f, Conv, SPPF, Detect, v8DetectionLoss), `configs/t_yolov8.yaml`
- Used by: `train.py`, `val.py`, `smoke_test.py`

**Data Layer:**
- Purpose: Load temporal frame triplets, tile images for training, apply temporally-consistent augmentations
- Location: `utils/`
- Contains: `TemporalDataset`, `temporal_collate_fn`, `TemporalAugmentor`, `FrameRegistrar`, `ECCRegistrar`
- Depends on: `cv2`, `numpy`, `torch.utils.data.Dataset`
- Used by: `train.py`, `val.py`

**Configuration Layer:**
- Purpose: Store model architecture specs and training hyperparameters
- Location: `configs/`
- Contains: `t_yolov8.yaml` (backbone/head definition), `t2_yolov8.yaml` (training config), `visdrone.yaml` (dataset paths and class names)
- Depends on: nothing (pure YAML)
- Used by: `models/t_yolov8.py`, `models/t2_yolov8.py`, `train.py`, `val.py`

**Entry Point Layer:**
- Purpose: CLI scripts that wire together model, dataset, optimizer, and evaluation
- Location: project root
- Contains: `train.py`, `val.py`, `smoke_test.py`
- Depends on: all layers above
- Used by: end user / researcher directly

**Dataset Preparation Layer:**
- Purpose: Convert raw VisDrone annotations to YOLO format and organize into sequence folders
- Location: `data/`, `scripts/`
- Contains: `data/visdrone_converter.py`, `scripts/convert_visdrone.py`, `scripts/download_visdrone.sh`
- Depends on: `cv2`, `numpy`
- Used by: offline preprocessing only (not imported by training code)

---

## Data Flow

**Training Flow:**

1. `train.py` loads `configs/t2_yolov8.yaml` (or `t_yolov8.yaml`) and merges CLI overrides into a `cfg` dict
2. `build_dataset()` constructs `TemporalDataset` pointing at `data/VisDrone/sequences/train/`
3. `TemporalDataset._index_sequences()` walks sequence folders and expands annotated frames into `(frame-triplet, tile-position)` sample pairs (320×320 tiles at training time)
4. Per `__getitem__`: three grayscale frames are loaded → support frames aligned to key frame via `ECCRegistrar` or `FrameRegistrar` → tile crop applied to all three frames simultaneously → `TemporalAugmentor` applies spatially-identical transforms across all three frames → `X_app` (3×H×W float32) and `X_mot` (2×H×W float32) tensors produced
5. `temporal_collate_fn` batches samples and prepends batch-index to label tensors: `labels` shape `(N_total, 6)` = `[batch_idx, cls, cx, cy, w, h]`
6. `model.loss(batch)` runs forward + `v8DetectionLoss` and returns `(loss_tensor(3,), loss_items)`
7. Gradient clipped to max-norm 10, optimizer stepped, cosine LR schedule applied per epoch
8. Every `val_period` epochs, `evaluate()` runs the full val loop and saves `best.pt` on mAP50 improvement

**Inference/Evaluation Flow:**

1. `val.py` loads checkpoint → `model.load_state_dict(state)`
2. `TemporalDataset` built without tiling (full-resolution, `tile_size=None`) and without augmentation
3. For T2: `model.predict(x_app, x_mot)` → `T2YOLOv8._extract_feats()` → `app_backbone + mot_backbone` → channel-wise concat at P3/P4/P5 → `_FPNNeck` → `Detect` head → raw `(B, 4+nc, A)` tensor
4. `decode_predictions()` applies sigmoid to class scores, filters by `conf_thres`, converts xywh→xyxy, applies per-class NMS via `torchvision.ops.nms`
5. `match_predictions()` matches detections to ground truth at 10 IoU thresholds (0.50–0.95), accumulates TP flags
6. `ultralytics.utils.metrics.ap_per_class` computes final mAP50, mAP50-95, precision, recall
7. Tiny-object sub-metrics computed separately for three area bins: Tiny1 (<32×32px), Tiny2 (32×32–64×64px), Tiny3 (64×64–128×128px)

**Two-Stream Fusion Flow (T2YOLOv8):**

```
X_app (B,3,H,W) ──► app_backbone (_BackboneExtractor) ──► app_p3, app_p4, app_p5
                                                                │
X_mot (B,2,H,W) ──► mot_backbone (_BackboneExtractor) ──► mot_p3, mot_p4, mot_p5
                                                                │
                    torch.cat([app_pN, mot_pN], dim=1) ──► fused_p3, fused_p4, fused_p5
                                                                │
                                                         _FPNNeck (top-down + bottom-up)
                                                                │
                                                         Detect head (P3/P4/P5)
```

**State Management:**
- All mutable state is held in `nn.Module` parameters; no global state
- `T2YOLOv8.criterion` is lazily initialized on first `loss()` call
- Checkpoint dict contains `{epoch, model (state_dict), optimizer (state_dict), cfg, map50}`

---

## Key Abstractions

**TYOLOv8** (`models/t_yolov8.py`):
- Purpose: Single-stream temporal detector; thin wrapper over `ultralytics.DetectionModel`
- Pattern: Subclass with custom `__init__` (injects scale into YAML config, enforces ch=3), `load_pretrained()` (shape-safe weight transfer), and a documented `forward()`
- Key behavior: temporal context encoded entirely in the 3-channel input `[f_{t-s}, f_t, f_{t+s}]`; no architectural changes to YOLOv8

**T2YOLOv8** (`models/t2_yolov8.py`):
- Purpose: Two-stream temporal detector with explicit motion stream
- Pattern: Custom `nn.Module` (not a subclass of DetectionModel) composed of `_BackboneExtractor × 2`, `_FPNNeck`, `Detect` head; exposes `predict(x_app, x_mot)` and `loss(batch)` explicitly; shims `self.model` property for `v8DetectionLoss` compatibility
- Key behavior: motion stream is always YOLOv8s (or Nano if main=Nano); concatenation fusion at P3/P4/P5 before FPN neck

**_BackboneExtractor** (`models/t2_yolov8.py`):
- Purpose: Extracts P3/P4/P5 feature maps from first 10 layers of a `DetectionModel`
- Pattern: Wraps `full_model.model.children()[:10]` into a `ModuleList`; fixed layer indices `_P3_IDX=4`, `_P4_IDX=6`, `_P5_IDX=9` (valid for all YOLOv8 scales)

**TemporalDataset** (`utils/temporal_dataset.py`):
- Purpose: PyTorch Dataset returning `{X_app, X_mot, labels}` for each annotated frame triplet
- Pattern: Index-then-expand pattern: `_index_sequences()` builds flat `self.samples` list during `__init__`; tile expansion at index time (not load time) means tile positions are pre-computed
- Key behavior: training tiles all three frames at the identical (y0,x0,y1,x1) crop to preserve temporal consistency; inference uses full-resolution resize

**TemporalAugmentor** (`utils/temporal_augmentation.py`):
- Purpose: Temporally-consistent spatial augmentation — every transform uses identical parameters for all three frames
- Pattern: Callable class; single affine matrix computed once and applied to all three frames; HSV jitter uses same `delta` across frames to preserve motion signal integrity
- Key novelties: `_temporal_mosaic()` (4-triplet mosaic, same layout across all frames), `_temporal_mixup()` (same alpha for all frames)

**Frame Registrars** (`utils/frame_registration.py`):
- Purpose: Align support frames to key frame to remove camera-motion artifacts from motion stream
- Pattern: Strategy pattern via `build_registrar(method)` factory; two implementations — `ECCRegistrar` (default, fast, 3-DoF euclidean) and `FrameRegistrar` (SIFT+RANSAC homography, robust)
- Both fall back to unregistered source frame on failure

---

## Entry Points

**`train.py`:**
- Location: `train.py` (project root)
- Triggers: `python train.py --config configs/t2_yolov8.yaml [overrides]`
- Responsibilities: parse args, load YAML config, build model + datasets + optimizer + scheduler, run epoch loop with warmup, periodic validation, checkpoint saving

**`val.py`:**
- Location: `val.py` (project root)
- Triggers: `python val.py --weights <path.pt> --config <yaml>` or imported as `from val import evaluate` in `train.py`
- Responsibilities: decode predictions, match to GT, accumulate TP/conf/class arrays, compute COCO-style AP + VisDrone tiny-object sub-metrics

**`smoke_test.py`:**
- Location: `smoke_test.py` (project root)
- Triggers: `python smoke_test.py [--device cpu]`
- Responsibilities: 7 integration tests covering both model variants (forward/loss/backward), tiling helpers, augmentor, collate function, and NMS decoding — no real data needed

**`data/visdrone_converter.py`:**
- Location: `data/visdrone_converter.py`
- Triggers: `python data/visdrone_converter.py --src /path/to/VisDrone --dst ./data/VisDrone`
- Responsibilities: offline preprocessing only; converts VisDrone annotations to YOLO format and organizes frames into `seq_{id}/frames/` + `seq_{id}/labels/` folder structure

---

## Error Handling

**Strategy:** Defensive + fail-fast with informative messages; no custom exception hierarchy.

**Patterns:**
- Missing config/checkpoint files raise `FileNotFoundError` with explicit path in message (e.g., `TYOLOv8.__init__`, `T2YOLOv8.__init__`)
- Shape-mismatched weights are silently skipped during `load_pretrained()` (graceful partial transfer); count of transferred/skipped layers logged via `ultralytics.utils.LOGGER`
- Frame registration failure (ECC convergence error, too few SIFT matches) returns the unregistered source frame rather than raising — allows training to continue on degraded data
- `TemporalDataset._index_sequences()` raises `RuntimeError` with actionable message if no labelled frames are found
- `decode_predictions()` returns `torch.zeros((0,6))` per image when no detections pass the confidence threshold
- `match_predictions()` returns empty arrays (not raises) when pred or GT count is zero

---

## Cross-Cutting Concerns

**Logging:** Via `ultralytics.utils.LOGGER` in model files; plain `print()` in training/evaluation scripts

**Validation:** Input shapes are implicitly validated by PyTorch tensor operations; label format (N,5) vs (N,6) distinction is enforced by convention in `temporal_collate_fn`

**Device Handling:** Device strings are resolved in `train.py`/`val.py` entry points; models/datasets are device-agnostic (`model.to(device)`, `batch.to(device)` pattern)

**Reproducibility:** `set_seed()` in `train.py` seeds `random`, `numpy`, and `torch` (including CUDA); seed defaults to 42 from config

**Temporal Consistency Invariant:** All spatial transforms (tile crop, affine augmentation, mosaic, mixup, flip) are always applied with identical parameters to all three frames in a triplet. This is the central correctness constraint of the entire pipeline.

---

*Architecture analysis: 2026-03-04*
