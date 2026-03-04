# Technology Stack

**Analysis Date:** 2026-03-04

## Language & Runtime

**Primary:**
- Python 3.12 (CPython) — all source code; confirmed by `.pyc` files in `__pycache__/` bearing `cpython-312` suffix

**Secondary:**
- Bash — `scripts/download_visdrone.sh` dataset download helper
- YAML — all configuration files in `configs/`

**Runtime Environment:**
- Conda-managed Python environment (`.vscode/settings.json` sets `python-envs.defaultEnvManager` to `ms-python.python:conda`)
- No `.python-version` or `.nvmrc` file; Python 3.12 inferred from bytecode artefacts

**Package Manager:**
- pip via Conda (no `setup.py`, `pyproject.toml`, or `setup.cfg`; dependencies declared in `requirements.txt`)
- Lockfile: **absent** — `requirements.txt` uses `>=` version pins only

---

## Frameworks & Libraries

**Core ML Framework:**
- `torch >= 2.0.0` (PyTorch) — model definition, training loop, loss computation, device management
  - Used in: `train.py`, `val.py`, `models/t_yolov8.py`, `models/t2_yolov8.py`, `utils/temporal_dataset.py`
- `torchvision >= 0.15.0` — `torchvision.ops.nms` used for per-class NMS in `val.py`

**Object Detection Framework:**
- `ultralytics >= 8.0.200` (YOLOv8 library) — backbone and detection head building blocks
  - `ultralytics.nn.tasks.DetectionModel` — base class for `TYOLOv8`
  - `ultralytics.nn.modules.{C2f, Conv, SPPF, Concat}` — neck architecture in `models/t2_yolov8.py`
  - `ultralytics.nn.modules.head.Detect` — detection head in `models/t2_yolov8.py`
  - `ultralytics.utils.loss.v8DetectionLoss` — YOLOv8 detection loss
  - `ultralytics.utils.torch_utils.initialize_weights` — weight initialisation
  - `ultralytics.utils.metrics.ap_per_class` — COCO-style AP evaluation in `val.py`
  - `ultralytics.nn.tasks.load_checkpoint` — pretrained weight loading
  - `ultralytics.utils.{LOGGER, DEFAULT_CFG}` — logging and default config

**Computer Vision:**
- `opencv-python >= 4.8.0` (cv2) — frame loading (IMREAD_GRAYSCALE), resize, warpAffine, warpPerspective, SIFT/ECC registration
  - Used extensively in: `utils/temporal_dataset.py`, `utils/temporal_augmentation.py`, `utils/frame_registration.py`, `data/visdrone_converter.py`

**Numerical / Scientific:**
- `numpy >= 1.24.0` — array operations, label processing, AP accumulation
- `scipy >= 1.11.0` — listed as dependency; not imported directly in reviewed source (likely pulled in by ultralytics)
- `Pillow >= 9.5.0` — listed as dependency; not imported directly in reviewed source (likely pulled in by ultralytics/torchvision)

**Augmentation:**
- `albumentations >= 1.3.0` — listed in `requirements.txt`; not directly imported in reviewed source. Custom temporal augmentation is implemented in `utils/temporal_augmentation.py` using numpy/cv2 instead

**Configuration & Utilities:**
- `PyYAML >= 6.0` — YAML config parsing in `train.py`, `val.py`, `models/t_yolov8.py`, `models/t2_yolov8.py`
- `tqdm >= 4.65.0` — listed as dependency; not visibly used in main scripts (likely used internally by ultralytics)
- `matplotlib >= 3.7.0` — listed as dependency; not used in reviewed source (likely used by ultralytics for plots)

---

## Dependencies

**Critical (directly used):**
| Package | Min Version | Role |
|---|---|---|
| `torch` | 2.0.0 | All deep learning ops, DataLoader, autograd |
| `torchvision` | 0.15.0 | `torchvision.ops.nms` for post-processing |
| `ultralytics` | 8.0.200 | YOLOv8 backbone, neck, head, loss, AP metrics |
| `opencv-python` | 4.8.0 | Image I/O, frame registration, augmentation |
| `numpy` | 1.24.0 | All numerical array work |
| `PyYAML` | 6.0 | Config file loading |

**Supporting (indirect/ultralytics deps):**
| Package | Min Version | Role |
|---|---|---|
| `scipy` | 1.11.0 | Scientific utilities (via ultralytics) |
| `Pillow` | 9.5.0 | Image handling (via ultralytics/torchvision) |
| `tqdm` | 4.65.0 | Progress bars (via ultralytics) |
| `matplotlib` | 3.7.0 | Plotting utilities (via ultralytics) |
| `albumentations` | 1.3.0 | Augmentation library (listed but not directly used) |

**Full dependency file:** `requirements.txt`

---

## Configuration

**Training config (primary):** `configs/t2_yolov8.yaml`
- Model type (`t` or `t2`), scale (`n/s/m/l/x`)
- Dataset path reference (`data: configs/visdrone.yaml`)
- Training hyperparameters: `epochs=300`, `batch_size=16`, `lr0=3.34e-3`, `momentum=0.937`, `weight_decay=2.5e-4`
- Augmentation flags: `mosaic_p`, `mixup_p`, `hflip_p`, `scale_range`, etc.
- Frame registration: `frame_registration: true`, `registration_method: ecc`
- Output: `project: runs`, `name: t2_yolov8x_visdrone`
- Device: `device: "0"` (CUDA index) or `"cpu"`

**Single-stream config:** `configs/t_yolov8.yaml`
- Architecture YAML defining backbone (Conv, C2f, SPPF) and FPN neck + Detect head
- Scale table: `n/s/m/l/x` with `[depth, width, max_channels]`

**Dataset config:** `configs/visdrone.yaml`
- Dataset root: `./data/VisDrone`
- Split paths: `sequences/train`, `sequences/val`, `sequences/test`
- 10 VisDrone classes: pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor

**Environment variables:** None required — all configuration is file-based via YAML

---

## Build & Tooling

**Linting / Formatting:**
- `ruff` (version 0.15.4) — cache present at `.ruff_cache/0.15.4/`; configured via `.ruff_cache/.gitignore`
- No `.eslintrc`, `.prettierrc`, or `black` configuration detected

**IDE:**
- VS Code with Python + Conda integration (`.vscode/settings.json`)

**Testing:**
- No pytest/unittest configuration file found
- Integration smoke test: `smoke_test.py` — manual run with `python smoke_test.py [--device cpu]`
  - Covers: forward pass, loss/backward, inference shape, tiling helpers, augmentor, collate fn, NMS decoding
  - Framework: plain Python `try/except` with pass/fail reporting; no test runner integration

**Data Tooling:**
- `scripts/download_visdrone.sh` — downloads VisDrone2019-DET from Google Drive (uses `gdown` or `wget`)
- `scripts/convert_visdrone.py` — thin CLI wrapper around `data/visdrone_converter.py`
- `data/visdrone_converter.py` — converts raw VisDrone annotations to YOLO temporal sequence format

**Pretrained Weights:**
- Auto-downloaded from Ultralytics servers at runtime via `ultralytics.nn.tasks.load_checkpoint`
- Naming convention: `yolov8{scale}.pt` (e.g. `yolov8x.pt`, `yolov8s.pt`)
- Checkpoints saved to: `runs/{name}/weights/` as `epoch{NNNN}.pt`, `best.pt`, `last.pt`

---

*Stack analysis: 2026-03-04*
