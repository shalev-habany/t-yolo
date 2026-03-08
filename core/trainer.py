"""
core/trainer.py

Trainer class — all training logic for T-YOLOv8 and T2-YOLOv8.

Public API
----------
    Trainer(cfg, app_weights, mot_weights)
    Trainer.run()

Module-level helpers
--------------------
    _cosine_lr(epoch, epochs, lrf)  -- picklable LR schedule function
    _TripletProvider                -- picklable triplet sampler for DataLoader workers
"""

from __future__ import annotations

import functools
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from models.t_yolov8 import TYOLOv8
from models.t2_yolov8 import T2YOLOv8
from utils.device import resolve_device
from utils.metrics import evaluate
from utils.temporal_augmentation import TemporalAugmentor
from utils.temporal_dataset import TemporalDataset, temporal_collate_fn


# ---------------------------------------------------------------------------
# Module-level helpers (must be at module level to be picklable)
# ---------------------------------------------------------------------------


def _cosine_lr(epoch: int, epochs: int, lrf: float) -> float:
    """Cosine annealing from 1.0 down to lrf over `epochs` steps."""
    return max(lrf, 0.5 * (1.0 + math.cos(math.pi * epoch / epochs)))


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

        for _ in range(10):
            s = self.samples[random.randint(0, len(self.samples) - 1)]
            f_pre = cv2.imread(str(s["pre_path"]), cv2.IMREAD_GRAYSCALE)
            f_key = cv2.imread(str(s["key_path"]), cv2.IMREAD_GRAYSCALE)
            f_post = cv2.imread(str(s["post_path"]), cv2.IMREAD_GRAYSCALE)
            if f_pre is None or f_key is None or f_post is None:
                # File unreadable (NFS hiccup, corrupt file, etc.) — try another sample
                continue
            if f_pre.size == 0 or f_key.size == 0 or f_post.size == 0:
                continue
            labels = _load_labels(s["label_path"])
            return f_pre, f_key, f_post, labels

        raise RuntimeError(
            "_TripletProvider: failed to load a valid triplet after 10 attempts. "
            "Check that frame files are accessible from DataLoader worker processes."
        )


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """
    Encapsulates all training logic for T-YOLOv8 / T2-YOLOv8.

    Args:
        cfg:         Merged config dict (YAML + CLI overrides already applied).
        app_weights: Path to appearance-stream pretrained weights, or None.
        mot_weights: Path to motion-stream pretrained weights, or None.
    """

    def __init__(
        self,
        cfg: dict,
        app_weights: str | None = None,
        mot_weights: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.app_weights = app_weights
        self.mot_weights = mot_weights

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute the full training loop."""
        cfg = self.cfg
        self._set_seed(cfg.get("seed", 42))

        # Device
        device_str = str(cfg.get("device", "0"))
        device = resolve_device(device_str)
        print(f"Device: {device}")

        # Dataset config
        with open(cfg["data"]) as f:
            data_cfg = yaml.safe_load(f)
        nc = data_cfg["nc"]
        data_root = Path(data_cfg["path"])
        train_root = data_root / data_cfg["train"]
        val_root = data_root / data_cfg["val"]

        # Model
        model_type = cfg.get("model", "t2")
        scale = cfg.get("scale", "x")
        app_weights = self.app_weights or cfg.get("app_weights")
        mot_weights = self.mot_weights or cfg.get("mot_weights")

        model = self._build_model(
            model_type=model_type,
            scale=scale,
            nc=nc,
            app_weights=app_weights,
            mot_weights=mot_weights,
            verbose=True,
        )
        model = model.to(device)

        # Datasets & loaders
        train_ds = self._build_dataset(
            train_root, cfg, split="train", model_type=model_type
        )
        val_ds = self._build_dataset(val_root, cfg, split="val", model_type=model_type)

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

        # Optimiser + scheduler
        optimizer = self._build_optimizer(model, cfg)
        epochs = cfg["epochs"]
        scheduler = self._build_scheduler(optimizer, cfg, epochs)

        warmup_epochs = cfg.get("warmup_epochs", 3.0)
        total_warmup_steps = max(round(warmup_epochs * len(train_loader)), 100)
        val_period = cfg.get("val_period", 10)

        # Output directory
        save_dir = Path(cfg.get("project", "runs")) / cfg.get("name", "exp")
        (save_dir / "weights").mkdir(parents=True, exist_ok=True)
        print(f"Saving to: {save_dir}")

        names: dict[int, str] = data_cfg.get("names", {i: str(i) for i in range(nc)})

        # Training loop
        best_map50 = 0.0
        global_step = 0

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            t0 = time.time()

            for batch_idx, batch in enumerate(train_loader):
                # Move tensors to device
                batch["img"] = batch["X_app"].to(device)
                if model_type == "t2":
                    batch["X_mot"] = batch["X_mot"].to(device)
                batch["labels"] = batch["labels"].to(device)

                # Reformat labels for ultralytics loss convention
                labels = batch["labels"]
                batch["cls"] = labels[:, 1:2]
                batch["bboxes"] = labels[:, 2:6]
                batch["batch_idx"] = labels[:, 0]
                tile_sz = cfg.get("tile_size", 320)
                batch["img_size"] = torch.tensor([tile_sz, tile_sz])

                # Warmup
                if global_step < total_warmup_steps:
                    self._warmup_lr(optimizer, global_step, total_warmup_steps, cfg)

                optimizer.zero_grad()
                loss, loss_items = model.loss(batch)
                loss = loss.sum()
                loss.backward()
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

            # Periodic validation
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

            # Save periodic checkpoint
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

            # Save best
            if do_val and map50 > best_map50:
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
        print(
            f"\nTraining complete. Best checkpoint: {save_dir / 'weights' / 'best.pt'}"
        )
        print(f"Best mAP50: {best_map50:.4f}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _build_model(
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
                app_weights=app_weights,
                mot_weights=mot_weights,
            )
        else:
            raise ValueError(f"Unknown model type '{model_type}'. Use 't' or 't2'.")
        return model

    @staticmethod
    def _build_dataset(
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

    @staticmethod
    def _build_optimizer(model: nn.Module, cfg: dict) -> SGD:
        """Build SGD optimiser with per-group weight decay (bias / BN excluded)."""
        g_wd, g_bias, g_bn = [], [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
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

    @staticmethod
    def _build_scheduler(optimizer: SGD, cfg: dict, epochs: int) -> LambdaLR:
        """Cosine LR schedule — uses module-level _cosine_lr for picklability."""
        lrf = cfg.get("lrf", 0.01)
        fn = functools.partial(_cosine_lr, epochs=epochs, lrf=lrf)
        return LambdaLR(optimizer, lr_lambda=fn)

    @staticmethod
    def _warmup_lr(
        optimizer: SGD,
        step: int,
        total_warmup_steps: int,
        cfg: dict,
    ) -> None:
        """Apply linear warmup to lr and momentum during first warmup_steps."""
        frac = min(step / max(total_warmup_steps, 1), 1.0)
        for i, pg in enumerate(optimizer.param_groups):
            if i == 0:
                pg["lr"] = cfg.get("warmup_bias_lr", 0.1) * frac
            else:
                pg["lr"] = cfg["lr0"] * frac
            if "momentum" in pg:
                pg["momentum"] = (
                    cfg.get("warmup_momentum", 0.8)
                    + (cfg["momentum"] - cfg.get("warmup_momentum", 0.8)) * frac
                )
