"""
train.py — thin CLI for training T-YOLOv8 / T2-YOLOv8 on VisDrone.

All training logic lives in core/trainer.py.

Usage:
    # Single-stream T-YOLOv8 (nano)
    python train.py --config configs/t_yolov8.yaml --scale n

    # Two-stream T2-YOLOv8 (nano)
    python train.py --config configs/t2_yolov8.yaml --scale n

    # T2: pass a pretrained T-YOLOv8 checkpoint for the appearance stream
    python train.py --config configs/t2_yolov8.yaml --scale n \\
        --app-weights runs/t_yolov8n_visdrone/weights/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))

from core.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train T-YOLOv8 / T2-YOLOv8 on VisDrone"
    )
    parser.add_argument(
        "--config",
        default="configs/t2_yolov8.yaml",
        help="Path to training config YAML (default: configs/t2_yolov8.yaml)",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Path to dataset config YAML (e.g. configs/visdrone.yaml). Overrides config.",
    )
    parser.add_argument(
        "--model",
        choices=["t", "t2"],
        default=None,
        help="Model type: 't' (single-stream) or 't2' (two-stream). Overrides config.",
    )
    parser.add_argument(
        "--scale",
        choices=["n", "s", "m", "l", "x"],
        default=None,
        help="Model scale. Overrides config.",
    )
    parser.add_argument(
        "--app-weights",
        default=None,
        help="Path to pretrained appearance stream weights. "
        "For T: COCO YOLOv8 .pt (auto-downloaded if None). "
        "For T2: trained T-YOLOv8 checkpoint.",
    )
    parser.add_argument(
        "--mot-weights",
        default=None,
        help="Path to pretrained motion stream weights (T2 only). "
        "Defaults to auto-downloaded yolov8s.pt.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs. Overrides config.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size. Overrides config.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="CUDA device index (e.g. '0', '1') or 'cpu'. Overrides config.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Apply CLI overrides
    if args.data is not None:
        cfg["data"] = args.data
    if args.model is not None:
        cfg["model"] = args.model
    if args.scale is not None:
        cfg["scale"] = args.scale
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.device is not None:
        cfg["device"] = args.device

    Trainer(cfg, app_weights=args.app_weights, mot_weights=args.mot_weights).run()
