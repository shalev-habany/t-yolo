# Directory Structure

**Analysis Date:** 2026-03-04

## Layout

```
T-YOLO/                          # Project root
├── train.py                     # Training entry point (CLI)
├── val.py                       # Evaluation entry point (CLI + importable evaluate())
├── smoke_test.py                # Integration smoke tests (no real data needed)
├── requirements.txt             # Python dependencies
│
├── models/                      # Neural network architectures
│   ├── __init__.py              # Re-exports TYOLOv8, T2YOLOv8
│   ├── t_yolov8.py              # Single-stream temporal model
│   └── t2_yolov8.py             # Two-stream temporal model
│
├── utils/                       # Data pipeline utilities
│   ├── __init__.py
│   ├── temporal_dataset.py      # TemporalDataset + collate function + tile helpers
│   ├── temporal_augmentation.py # TemporalAugmentor (mosaic, mixup, spatial transforms)
│   └── frame_registration.py   # ECCRegistrar + FrameRegistrar (SIFT) + factory
│
├── configs/                     # YAML configuration files
│   ├── t_yolov8.yaml            # Backbone/head architecture definition (YOLOv8 graph)
│   ├── t2_yolov8.yaml           # Training hyperparameters + model selection
│   └── visdrone.yaml            # Dataset paths and class names
│
├── data/                        # Dataset preprocessing scripts
│   └── visdrone_converter.py    # VisDrone → YOLO format converter (offline use only)
│
├── scripts/                     # Shell and helper scripts
│   ├── download_visdrone.sh     # Download VisDrone dataset
│   └── convert_visdrone.py      # Alternative converter entry point
│
├── .planning/                   # GSD planning documents (not part of ML pipeline)
│   └── codebase/
│
└── Corsel_Exploiting_Temporal_Context_...pdf   # Source paper (reference)
```

**Runtime-generated (not committed):**
```
runs/                            # Training outputs (created by train.py)
  └── <name>/
      └── weights/
          ├── best.pt            # Best checkpoint by mAP50
          ├── last.pt            # Final epoch checkpoint
          └── epoch<NNNN>.pt     # Periodic checkpoints

data/VisDrone/                   # Converted dataset (created by visdrone_converter.py)
  └── sequences/
      ├── train/
      │   └── seq_<id>/
      │       ├── frames/        # .jpg or .png images
      │       └── labels/        # .txt YOLO-format annotations (key frames only)
      ├── val/
      └── test/
```

---

## Directory Purposes

**`models/`:**
- Purpose: All neural network architecture code
- Contains: Two concrete model classes and their internal submodules (`_BackboneExtractor`, `_FPNNeck` are defined inside `t2_yolov8.py`, not separate files)
- Key files: `t_yolov8.py` (TYOLOv8), `t2_yolov8.py` (T2YOLOv8)
- Convention: One model variant per file; internal helper classes kept in the same file as the model that uses them

**`utils/`:**
- Purpose: Data pipeline — loading, augmentation, registration
- Contains: One Python module per major concern (dataset, augmentation, registration)
- Key files: `temporal_dataset.py` (central data class + all tile/label helpers), `temporal_augmentation.py` (TemporalAugmentor), `frame_registration.py` (registrar implementations + factory)
- Note: Module-level private helpers (prefixed `_`) are defined in the same file as their consumer rather than a shared helpers module

**`configs/`:**
- Purpose: Externalize all hyperparameters and architecture specs
- `t_yolov8.yaml` — YOLOv8 backbone+head graph definition consumed by `ultralytics.DetectionModel`; scale depth/width/max_channels table; only model architecture, no training params
- `t2_yolov8.yaml` — Training run config: model type, scale, optimizer params, augmentation probs, registration settings, output paths; this is the primary config file passed to `train.py`/`val.py`
- `visdrone.yaml` — Dataset layout and class names; referenced from `t2_yolov8.yaml` as `data:` key

**`data/`:**
- Purpose: Offline dataset preparation only; not imported during training/evaluation
- Convention: Preprocessing scripts that produce the `data/VisDrone/sequences/` layout consumed by `TemporalDataset`

**`scripts/`:**
- Purpose: Shell and helper scripts for environment setup and dataset acquisition
- Not imported by any training code

---

## Key File Locations

**Entry Points:**
- `train.py`: Full training loop with argparse CLI; `if __name__ == "__main__"` guard
- `val.py`: Evaluation loop; also importable — `train.py` does `from val import evaluate`
- `smoke_test.py`: Self-contained integration test runner; no test framework dependency

**Model Definitions:**
- `models/t_yolov8.py`: `TYOLOv8` class — single-stream model
- `models/t2_yolov8.py`: `T2YOLOv8`, `_BackboneExtractor`, `_FPNNeck`, `_load_backbone_weights`

**Data Pipeline:**
- `utils/temporal_dataset.py`: `TemporalDataset`, `temporal_collate_fn`, `_compute_tile_positions`, `_clip_labels_to_tile`, `_load_gray`, `_load_labels`
- `utils/temporal_augmentation.py`: `TemporalAugmentor`, `_build_affine_matrix`, `_apply_affine_to_labels`
- `utils/frame_registration.py`: `ECCRegistrar`, `FrameRegistrar`, `build_registrar` (factory)

**Configuration:**
- `configs/t_yolov8.yaml`: Model architecture spec (backbone + head layers, scales table)
- `configs/t2_yolov8.yaml`: Training config (primary YAML passed to `--config`)
- `configs/visdrone.yaml`: Dataset root path, split names, class names

**Public API (`models/__init__.py`):**
```python
from models import TYOLOv8, T2YOLOv8
```

---

## Naming Conventions

**Files:**
- `snake_case.py` for all Python files
- Module name matches the paper concept it implements (`t_yolov8`, `t2_yolov8`, `temporal_dataset`, `temporal_augmentation`, `frame_registration`)
- Config files use `snake_case.yaml`; named after the model or dataset they configure

**Classes:**
- `PascalCase` for all classes
- Model classes: `TYOLOv8`, `T2YOLOv8` (abbreviation-style, matching paper naming)
- Internal/private classes: prefixed with `_` (`_BackboneExtractor`, `_FPNNeck`, `_Registrar`, `_Augmentor`)
- Protocol classes: prefixed with `_` (`_Registrar`, `_Augmentor` in `temporal_dataset.py`)

**Functions:**
- `snake_case` for all functions
- Private module-level helpers: prefixed with `_` (`_load_gray`, `_load_labels`, `_compute_tile_positions`, `_clip_labels_to_tile`, `_xywh2xyxy`, `_xyxy_iou`, `_gt_to_xyxy`, `_build_affine_matrix`, `_apply_affine_to_labels`, `_clip_boxes`, `_value_jitter_delta`)
- Public factory: `build_registrar()` (no underscore)
- Builder helpers in `train.py`: `build_model()`, `build_dataset()`, `build_optimizer()`, `build_scheduler()`

**Variables / Tensors:**
- Tensor inputs: `X_app` (appearance stream, 3-channel), `X_mot` (motion stream, 2-channel) — uppercase to match paper notation
- Feature maps: `p3`, `p4`, `p5` (lowercase) for neck inputs/outputs; `app_p3`, `mot_p3` for stream-prefixed variants
- Frame arrays: `f_pre`, `f_key`, `f_post` (paper notation for `f_{t-s}`, `f_t`, `f_{t+s}`)
- Config dicts: `cfg` (training config), `data_cfg` (dataset config)
- Checkpoint dict keys: `"model"`, `"optimizer"`, `"epoch"`, `"cfg"`, `"map50"`

**Constants:**
- `UPPER_CASE` for module-level constants (`_PRETRAIN_MAP`, `_MOTION_SCALE`, `_TINY_AREA_THRESHOLDS`, `CAT_MAP`, `SPLITS`)

---

## Where to Add New Code

**New model architecture:**
- Add `models/<model_name>.py` following the pattern of `models/t_yolov8.py` or `models/t2_yolov8.py`
- Export from `models/__init__.py`
- Add a corresponding `configs/<model_name>.yaml` if architecture changes are needed
- Add `build_model()` branch in `train.py`

**New augmentation:**
- Add method to `TemporalAugmentor` in `utils/temporal_augmentation.py`
- Call it in `TemporalAugmentor.__call__()` with probability gating pattern: `if self.p > 0 and random.random() < self.p`
- Add corresponding config key to `configs/t2_yolov8.yaml` and wire it in `train.py::build_dataset()`

**New evaluation metric:**
- Add computation inside `val.py::evaluate()` after the existing `_compute_ap()` calls
- Add key to returned `results` dict

**New dataset:**
- Add a converter script in `data/` following `data/visdrone_converter.py`
- Create a new `configs/<dataset>.yaml` following `configs/visdrone.yaml`
- Point `data:` in training config to the new dataset YAML

**New frame registration method:**
- Add class with `.register(src, ref) -> np.ndarray` method in `utils/frame_registration.py`
- Add branch in `build_registrar()` factory function
- Use method name string in `configs/t2_yolov8.yaml` under `registration_method:`

---

## Special Directories

**`.planning/`:**
- Purpose: GSD planning and mapping documents
- Generated: No (written by GSD tooling)
- Committed: Yes (planning artifacts)

**`__pycache__/`:**
- Purpose: Python bytecode cache
- Generated: Yes (automatically by Python)
- Committed: No

**`.ruff_cache/`:**
- Purpose: Ruff linter cache
- Generated: Yes (by ruff linter)
- Committed: No

**`runs/`:**
- Purpose: Training output directory — checkpoints and logs
- Generated: Yes (by `train.py`)
- Committed: No (too large; contains model weights)

**`data/VisDrone/`:**
- Purpose: Processed dataset in temporal sequence format
- Generated: Yes (by `data/visdrone_converter.py`)
- Committed: No (dataset files)

---

*Structure analysis: 2026-03-04*
