"""
train.py

Training entry point for T-YOLOv8 and T2-YOLOv8.

Paper: "Exploiting Temporal Context for Tiny Object Detection"
       Corsel et al., WACVW 2023

Training strategy (paper §4.1):
    - SGD, lr=3.34e-3, weight_decay=2.5e-4, 300 epochs
    - Cosine LR schedule
    - Input: 320×320 tiled crops during training, full-resolution at inference
    - Pretrained from COCO (YOLOv8 weights)
    - For T2: transfer T-YOLOv8 weights to main stream; motion stream from YOLOv8s COCO

Usage:
    # Single-stream
    python train.py --model t --scale x --config configs/t2_yolov8.yaml

    # Two-stream (requires pretrained T-YOLOv8 checkpoint for app stream)
    python train.py --model t2 --scale x --config configs/t2_yolov8.yaml \\
        --app-weights runs/t_yolov8x_visdrone/weights/best.pt
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

# Project root on path
sys.path.insert(0, str(Path(__file__).parent))

from models.t_yolov8 import TYOLOv8
from models.t2_yolov8 import T2YOLOv8
from utils.temporal_augmentation import TemporalAugmentor
from utils.temporal_dataset import TemporalDataset, temporal_collate_fn
from val import evaluate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(
    model_type: str,
    scale: str,
    nc: int,
    app_weights: str | None,
    mot_weights: str | None,
    verbose: bool = True,
) -> TYOLOv8 | T2YOLOv8:
    """Build and optionally load pretrained weights."""
    if model_type == "t":
        model = TYOLOv8(scale=scale, nc=nc, verbose=verbose)
        if app_weights is not None:
            model.load_pretrained(app_weights)
        else:
            model.load_pretrained()  # COCO weights
    elif model_type == "t2":
        model = T2YOLOv8(scale=scale, nc=nc, verbose=verbose)
        model.load_pretrained(
            app_weights=app_weights,  # T-YOLOv8 checkpoint (may be None)
            mot_weights=mot_weights,  # COCO YOLOv8s.pt (auto-download if None)
        )
    else:
        raise ValueError(f"Unknown model type '{model_type}'. Use 't' or 't2'.")
    return model


class _TripletProvider:
    """
    Picklable callable that returns a random (f_pre, f_key, f_post, labels)
    triplet from a TemporalDataset's sample list.

    A plain closure cannot be pickled by Python's spawn-based multiprocessing
    (the default on macOS), so we use a top-level class instead.
    """

    def __init__(self, samples: list) -> None:
        self.samples = samples

    def __call__(self):
        import cv2
        from utils.temporal_dataset import _load_labels

        s = self.samples[random.randint(0, len(self.samples) - 1)]
        f_pre = cv2.imread(str(s["pre_path"]), cv2.IMREAD_GRAYSCALE)
        f_key = cv2.imread(str(s["key_path"]), cv2.IMREAD_GRAYSCALE)
        f_post = cv2.imread(str(s["post_path"]), cv2.IMREAD_GRAYSCALE)
        labels = _load_labels(s["label_path"])
        return f_pre, f_key, f_post, labels


def build_dataset(
    split_root: str | Path,
    cfg: dict,
    split: str = "train",
    model_type: str = "t",
) -> TemporalDataset:
    """Build TemporalDataset for a given split."""
    augment = (split == "train") and cfg.get("augment", True)
    two_stream = model_type == "t2"

    img_size = (cfg["img_size"], cfg["img_size"])

    # At training time: use 320×320 tiled crops (paper §3.3).
    # At val/test time: resize the full frame to img_size.
    tile_size = None
    tile_overlap = cfg.get("tile_overlap", 0.05)
    if split == "train":
        ts = cfg.get("tile_size", 320)
        tile_size = (ts, ts)

    dataset = TemporalDataset(
        split_root=split_root,
        temporal_shift=cfg.get("temporal_shift", 3),
        img_size=img_size,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        register_frames=cfg.get("frame_registration", True),
        registration_method=cfg.get("registration_method", "ecc"),
        two_stream=two_stream,
        augment=augment,
    )

    # Wire up Temporal Mosaic / MixUp: augmentor needs a triplet provider.
    # Use a top-level picklable class so DataLoader workers (spawn) can pickle it.
    if augment and (cfg.get("mosaic_p", 0.0) > 0 or cfg.get("mixup_p", 0.0) > 0):
        augmentor = TemporalAugmentor(
            img_size=tile_size or img_size,
            hflip_p=cfg.get("hflip_p", 0.5),
            vflip_p=cfg.get("vflip_p", 0.0),
            scale_range=tuple(cfg.get("scale_range", [0.5, 1.5])),
            translate_frac=cfg.get("translate_frac", 0.1),
            mosaic_p=cfg.get("mosaic_p", 0.5),
            mixup_p=cfg.get("mixup_p", 0.1),
            triplet_provider=_TripletProvider(dataset.samples),
        )
        dataset.augmentor = augmentor  # type: ignore[assignment]

    return dataset


def build_optimizer(
    model: nn.Module,
    cfg: dict,
) -> SGD:
    """Build SGD optimiser with per-group weight decay (bias / BN excluded)."""
    g_wd, g_bias, g_bn = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # BN parameters (weight and bias) are identified by the module type, not
        # by name substring, so that ".bn.bias" goes to g_bn (no weight decay)
        # rather than g_bias.  Walk the module tree to find the parent module.
        *parent_parts, param_name = name.split(".")
        parent = model
        for part in parent_parts:
            parent = getattr(parent, part)
        if isinstance(parent, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            g_bn.append(param)
        elif param_name == "bias":
            g_bias.append(param)
        else:
            g_wd.append(param)

    optimizer = SGD(
        g_bias,
        lr=cfg["lr0"],
        momentum=cfg["momentum"],
        nesterov=True,
    )
    optimizer.add_param_group({"params": g_bn, "weight_decay": 0.0})
    optimizer.add_param_group({"params": g_wd, "weight_decay": cfg["weight_decay"]})
    return optimizer


def build_scheduler(
    optimizer: SGD,
    cfg: dict,
    epochs: int,
) -> LambdaLR:
    """Cosine LR schedule with warmup (matches ultralytics default)."""
    lrf = cfg.get("lrf", 0.01)

    def lr_lambda(epoch: int) -> float:
        # Cosine annealing: 1.0 → lrf
        return max(lrf, 0.5 * (1.0 + math.cos(math.pi * epoch / epochs)))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def warmup_lr(
    optimizer: SGD,
    step: int,
    total_warmup_steps: int,
    cfg: dict,
) -> None:
    """Apply linear warmup to lr and momentum during first warmup_steps."""
    frac = min(step / max(total_warmup_steps, 1), 1.0)
    for i, pg in enumerate(optimizer.param_groups):
        # Warmup bias lr
        if i == 0:
            pg["lr"] = cfg.get("warmup_bias_lr", 0.1) * frac
        else:
            pg["lr"] = cfg["lr0"] * frac
        if "momentum" in pg:
            pg["momentum"] = (
                cfg.get("warmup_momentum", 0.8)
                + (cfg["momentum"] - cfg.get("warmup_momentum", 0.8)) * frac
            )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(cfg: dict, args: argparse.Namespace) -> None:
    set_seed(cfg.get("seed", 42))

    # --- Device ---
    device_str = str(cfg.get("device", "0"))
    if device_str == "cpu":
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{device_str}")
    else:
        print("CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    print(f"Device: {device}")

    # --- Load dataset config ---
    with open(cfg["data"]) as f:
        data_cfg = yaml.safe_load(f)
    nc = data_cfg["nc"]
    data_root = Path(data_cfg["path"])
    train_root = data_root / data_cfg["train"]
    val_root = data_root / data_cfg["val"]

    # --- Model ---
    model_type = cfg.get("model", "t2")
    scale = cfg.get("scale", "x")
    app_weights = args.app_weights or cfg.get("app_weights")
    mot_weights = args.mot_weights or cfg.get("mot_weights")

    model = build_model(
        model_type=model_type,
        scale=scale,
        nc=nc,
        app_weights=app_weights,
        mot_weights=mot_weights,
        verbose=True,
    )
    model = model.to(device)

    # --- Datasets & loaders ---
    train_ds = build_dataset(train_root, cfg, split="train", model_type=model_type)
    val_ds = build_dataset(val_root, cfg, split="val", model_type=model_type)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg.get("workers", 4),
        collate_fn=temporal_collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"] * 2,
        shuffle=False,
        num_workers=cfg.get("workers", 4),
        collate_fn=temporal_collate_fn,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    # --- Optimiser + scheduler ---
    optimizer = build_optimizer(model, cfg)
    epochs = cfg["epochs"]
    scheduler = build_scheduler(optimizer, cfg, epochs)

    warmup_epochs = cfg.get("warmup_epochs", 3.0)
    total_warmup_steps = max(round(warmup_epochs * len(train_loader)), 100)

    # How often to run full validation (mAP) during training.
    # Default: every 10 epochs, always on final epoch.
    val_period = cfg.get("val_period", 10)

    # --- Output directory ---
    save_dir = Path(cfg.get("project", "runs")) / cfg.get("name", "exp")
    (save_dir / "weights").mkdir(parents=True, exist_ok=True)
    print(f"Saving to: {save_dir}")

    # --- Dataset class names ---
    names: dict[int, str] = data_cfg.get("names", {i: str(i) for i in range(nc)})

    # --- Training loop ---
    best_map50 = 0.0
    global_step = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            # Move tensors to device
            batch["img"] = batch["X_app"].to(device)  # (B, 3, H, W)
            if model_type == "t2":
                batch["X_mot"] = batch["X_mot"].to(device)  # (B, 2, H, W)
            batch["labels"] = batch["labels"].to(
                device
            )  # (N, 6): [batch_idx, cls, cx, cy, w, h]

            # Reformat labels for ultralytics loss convention
            # ultralytics expects batch["cls"] (N,) and batch["bboxes"] (N, 4)
            labels = batch["labels"]  # (N, 6)
            batch["cls"] = labels[:, 1:2]  # (N, 1)
            batch["bboxes"] = labels[:, 2:6]  # (N, 4)
            batch["batch_idx"] = labels[:, 0]  # (N,)
            tile_sz = cfg.get("tile_size", 320)
            batch["img_size"] = torch.tensor([tile_sz, tile_sz])

            # Warmup
            if global_step < total_warmup_steps:
                warmup_lr(optimizer, global_step, total_warmup_steps, cfg)

            optimizer.zero_grad()

            loss, loss_items = model.loss(batch)

            # loss is shape (3,) = [box, cls, dfl]; sum for scalar backward
            loss = loss.sum()

            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

            optimizer.step()
            global_step += 1
            epoch_loss += loss.item()

        scheduler.step()

        avg_loss = epoch_loss / max(len(train_loader), 1)
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[-1]["lr"]
        print(
            f"Epoch {epoch + 1:3d}/{epochs}  "
            f"loss={avg_loss:.4f}  "
            f"lr={lr_now:.2e}  "
            f"time={elapsed:.1f}s"
        )

        # --- Periodic validation (mAP50) ---
        do_val = (epoch + 1) % val_period == 0 or (epoch + 1) == epochs
        map50 = 0.0
        if do_val:
            val_results = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                nc=nc,
                names=names,
                model_type=model_type,
                img_size=cfg.get("img_size", 640),
            )
            map50 = val_results["mAP50"]
            map5095 = val_results["mAP50-95"]
            print(
                f"  Val  mAP50={map50:.4f}  mAP50-95={map5095:.4f}  "
                f"P={val_results['precision']:.4f}  R={val_results['recall']:.4f}"
            )

        # --- Save checkpoints ---
        save_period = cfg.get("save_period", 10)
        if (epoch + 1) % save_period == 0 or (epoch + 1) == epochs:
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "cfg": cfg,
                "map50": map50,
            }
            torch.save(ckpt, save_dir / "weights" / f"epoch{epoch + 1:04d}.pt")

        # --- Save best based on mAP50 (or loss proxy before first val) ---
        if do_val:
            if map50 > best_map50:
                best_map50 = map50
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "cfg": cfg,
                        "map50": map50,
                    },
                    save_dir / "weights" / "best.pt",
                )
                print(f"  Saved best checkpoint (mAP50={best_map50:.4f})")

    torch.save(
        {"epoch": epochs - 1, "model": model.state_dict(), "cfg": cfg},
        save_dir / "weights" / "last.pt",
    )
    print(f"\nTraining complete. Best checkpoint: {save_dir / 'weights' / 'best.pt'}")
    print(f"Best mAP50: {best_map50:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


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

    # CLI overrides
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

    train(cfg, args)
