"""
val.py — thin CLI for evaluating T-YOLOv8 / T2-YOLOv8 on VisDrone.

All evaluation logic lives in utils/metrics.py.

Usage:
    python val.py --weights runs/t_yolov8n_visdrone/weights/best.pt \\
                  --config configs/t_yolov8.yaml

    python val.py --weights runs/t2_yolov8x_visdrone/weights/best.pt \\
                  --config configs/t2_yolov8.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from models.t_yolov8 import TYOLOv8
from models.t2_yolov8 import T2YOLOv8
from utils.device import resolve_device
from utils.metrics import evaluate
from utils.temporal_dataset import TemporalDataset, temporal_collate_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate T-YOLOv8 / T2-YOLOv8 on VisDrone"
    )
    parser.add_argument(
        "--weights", required=True, help="Path to model checkpoint (.pt)"
    )
    parser.add_argument(
        "--config",
        default="configs/t2_yolov8.yaml",
        help="Training config YAML (model type, scale, dataset path, etc.)",
    )
    parser.add_argument("--model", choices=["t", "t2"], default=None)
    parser.add_argument("--scale", choices=["n", "s", "m", "l", "x"], default=None)
    parser.add_argument(
        "--split",
        default="val",
        choices=["val", "test"],
        help="Dataset split to evaluate",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--conf-thres", type=float, default=0.001)
    parser.add_argument("--iou-thres", type=float, default=0.65)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--img-size",
        type=int,
        default=None,
        help="Inference image size (overrides config)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.model:
        cfg["model"] = args.model
    if args.scale:
        cfg["scale"] = args.scale
    if args.device:
        cfg["device"] = args.device
    if args.img_size:
        cfg["img_size"] = args.img_size

    device = resolve_device(str(cfg.get("device", "0")))

    # Load dataset config
    with open(cfg["data"]) as f:
        data_cfg = yaml.safe_load(f)
    nc = data_cfg["nc"]
    names: dict[int, str] = data_cfg["names"]
    data_root = Path(data_cfg["path"])
    split_root = data_root / data_cfg[args.split]

    model_type = cfg.get("model", "t2")
    scale = cfg.get("scale", "x")

    # Build model
    if model_type == "t":
        model = TYOLOv8(scale=scale, nc=nc, verbose=False)
    else:
        model = T2YOLOv8(scale=scale, nc=nc, verbose=False)

    # Load checkpoint
    ckpt = torch.load(args.weights, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    print(f"Loaded weights from {args.weights}")

    # Dataset
    img_size = cfg.get("img_size", 640)
    dataset = TemporalDataset(
        split_root=split_root,
        temporal_shift=cfg.get("temporal_shift", 3),
        img_size=(img_size, img_size),
        register_frames=cfg.get("frame_registration", True),
        registration_method=cfg.get("registration_method", "ecc"),
        two_stream=(model_type == "t2"),
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=cfg.get("workers", 4),
        collate_fn=temporal_collate_fn,
        pin_memory=True,
    )
    print(f"Evaluating on {len(dataset)} samples from {split_root}")

    results = evaluate(
        model=model,
        loader=loader,
        device=device,
        nc=nc,
        names=names,
        model_type=model_type,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        img_size=img_size,
    )

    # Print results table
    print("\n" + "=" * 60)
    print(f"{'Metric':<20} {'Value':>10}")
    print("-" * 60)
    for k, v in results.items():
        print(f"{k:<20} {v:>10.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
