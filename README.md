# T-YOLO

Temporal YOLOv8 for tiny-object detection on VisDrone.
Implementation of Corsel et al., "Exploiting Temporal Context for Tiny Object Detection", WACVW 2023.

## Install

```bash
pip install -r requirements.txt
```

## Download VisDrone

```bash
bash scripts/download_visdrone.sh --dst data/raw_visdrone
```

## Convert VisDrone

```bash
# Full dataset root (all splits present under data/raw_visdrone)
python data/visdrone_converter.py --src data/raw_visdrone --dst data/VisDrone

# Single split only (e.g. val)
python data/visdrone_converter.py --src data/VisDrone2019-VID-val --dst data/VisDrone
```

## Train

```bash
# Single-stream T-YOLOv8 (nano)
python train.py --config configs/t_yolov8.yaml --scale n

# Two-stream T2-YOLOv8 (nano)
python train.py --config configs/t2_yolov8.yaml --scale n

# Override dataset path without editing YAML
python train.py --config configs/t_yolov8.yaml --data configs/visdrone.yaml --scale n

# T2: pass a pretrained T-YOLOv8 checkpoint for the appearance stream
python train.py --config configs/t2_yolov8.yaml --scale n \
    --app-weights runs/t_yolov8n_visdrone/weights/best.pt
```

## Validate

```bash
python val.py --weights runs/t_yolov8n_visdrone/weights/best.pt \
              --config configs/t_yolov8.yaml
```

## Smoke test (no real data required)

```bash
python smoke_test.py --device cpu
```
