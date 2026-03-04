# External Integrations

**Analysis Date:** 2026-03-04

## APIs & External Services

**Ultralytics Model Hub (implicit):**
- Service: Ultralytics pretrained weight server (COCO-trained YOLOv8 checkpoints)
- Triggered: at runtime when `load_pretrained()` is called with `weights_path=None`
- Mechanism: `ultralytics.nn.tasks.load_checkpoint(weights_path, device="cpu")` auto-downloads if the `.pt` file is not on disk
- Files used: `models/t_yolov8.py` (line ~122), `models/t2_yolov8.py` (line ~501)
- Weight filenames: `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`, `yolov8x.pt`
- No API key required; relies on ultralytics HTTP download

**Google Drive (dataset download only):**
- Service: Google Drive public file hosting
- Triggered: manually via `scripts/download_visdrone.sh`
- Mechanism: `gdown` CLI or `wget` with cookie handling
- File IDs:
  - Train split: `1a2oHjcEcwXP3DrptG-GQ2YGCV3-3n9bm`
  - Val split: `1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59`
  - Test-dev split: `1PFdW_VFSCfZ_sTSZAGjQdifF_Xd5mf0V`
- Not used during training or inference — download step only
- Requires `gdown` (`pip install gdown`) or system `wget`

---

## Databases

**None.** The project uses the local filesystem exclusively.

**Dataset storage:**
- Root: `./data/VisDrone/` (configurable via `configs/visdrone.yaml` `path:` key)
- Structure:
  ```
  data/VisDrone/
    sequences/
      train/  val/  test/
        seq_<id>/
          frames/   *.jpg
          labels/   *.txt  (YOLO format: cls cx cy w h)
  ```
- Populated by running `data/visdrone_converter.py` against the raw VisDrone download

**Checkpoint storage:**
- Root: `runs/{name}/weights/`
- Files: `epoch{NNNN}.pt`, `best.pt`, `last.pt`
- Format: Python dict serialised with `torch.save` — keys: `epoch`, `model` (state_dict), `optimizer` (state_dict), `cfg`, `map50`

---

## Auth Providers

**None.** No authentication is required for any component.

- Ultralytics weight download: unauthenticated public URLs
- VisDrone dataset: Google Drive public links (no OAuth); `gdown` handles the confirmation cookie internally
- No user accounts, sessions, or tokens

---

## Webhooks & Events

**None.** The project is a standalone research training/evaluation pipeline with no web server, event bus, or webhook endpoints.

---

## Environment Variables

**None required.** All configuration is file-based (YAML).

The following parameters that might typically be environment variables are instead set in YAML configs or passed via CLI:

| Setting | Location | How to Override |
|---|---|---|
| CUDA device | `configs/t2_yolov8.yaml` → `device: "0"` | `--device` CLI arg or edit YAML |
| Dataset path | `configs/visdrone.yaml` → `path:` | Edit YAML |
| Batch size | `configs/t2_yolov8.yaml` → `batch_size:` | `--batch-size` CLI arg |
| Pretrained weights path | `configs/t2_yolov8.yaml` → `app_weights:` / `mot_weights:` | `--app-weights` / `--mot-weights` CLI args |
| Output directory | `configs/t2_yolov8.yaml` → `project:` / `name:` | Edit YAML |
| Random seed | `configs/t2_yolov8.yaml` → `seed: 42` | Edit YAML |

**`.env` file:** Not present. No secrets or credentials are used.

---

## Third-Party Data Sources

**VisDrone2019-DET dataset:**
- Source: VisDrone workshop (ECCV/ICCV/CVPR), hosted on Google Drive
- Official site: https://github.com/VisDrone/VisDrone-Dataset
- Splits: train (6,471 images), val (548 images), test-dev (1,610 images)
- Classes: 10 (pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor)
- Not bundled — must be downloaded separately via `scripts/download_visdrone.sh`

**COCO-pretrained YOLOv8 weights:**
- Source: Ultralytics (auto-downloaded at first `load_pretrained()` call)
- Used for: initialising appearance stream (T-YOLOv8) and motion stream (YOLOv8s for T2-YOLOv8)
- Not versioned in repo — managed by ultralytics library

---

## Research Reference

**Paper:** "Exploiting Temporal Context for Tiny Object Detection" — Corsel et al., WACVW 2023
- Bundled as `Corsel_Exploiting_Temporal_Context_for_Tiny_Object_Detection_WACVW_2023_paper.pdf`
- Not an integration; included for reference only

---

*Integration audit: 2026-03-04*
