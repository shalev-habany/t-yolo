"""
utils/metrics.py

Evaluation helpers for T-YOLOv8 and T2-YOLOv8 on VisDrone.

Public API
----------
    decode_predictions  -- NMS + prediction parsing
    match_predictions   -- TP matching against ground-truth
    evaluate            -- full evaluation loop; returns metric dict

Internal helpers
----------------
    _xywh2xyxy, _xyxy_iou, _gt_to_xyxy, _gt_area_px
    _TINY_AREA_THRESHOLDS
    _compute_ap         -- module-level (was nested inside evaluate)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.ops import nms


# ---------------------------------------------------------------------------
# NMS + prediction parsing
# ---------------------------------------------------------------------------


def decode_predictions(
    raw: torch.Tensor,
    conf_thres: float = 0.001,
    iou_thres: float = 0.65,
    nc: int = 10,
) -> list[torch.Tensor]:
    """
    Decode the Detect head output and apply NMS per image.

    Args:
        raw:        (B, 4+nc, num_anchors) -- xywh (image coords) + class scores.
        conf_thres: Minimum confidence score to keep.
        iou_thres:  NMS IoU threshold.
        nc:         Number of classes.

    Returns:
        List of length B. Each element is a tensor (N, 6): [x1, y1, x2, y2, conf, cls].
    """
    # raw: (B, 4+nc, A) -> (B, A, 4+nc)
    preds = raw.permute(0, 2, 1).contiguous()  # (B, A, 4+nc)

    results = []
    for pred in preds:  # (A, 4+nc) per image
        # Class scores: apply sigmoid to convert logits -> probabilities
        cls_scores = pred[:, 4:].sigmoid()  # (A, nc)
        # Best class per anchor
        conf, cls_idx = cls_scores.max(dim=1)  # (A,)

        # Filter by confidence
        mask = conf > conf_thres
        if not mask.any():
            results.append(torch.zeros((0, 6), device=pred.device))
            continue

        boxes_xywh = pred[mask, :4]  # (K, 4) in xywh
        conf_k = conf[mask]
        cls_k = cls_idx[mask].float()

        # xywh -> xyxy
        boxes_xyxy = _xywh2xyxy(boxes_xywh)

        # Per-class NMS: offset boxes by class id so boxes of different classes
        # never suppress each other (standard torchvision approach).
        max_wh = 4096.0  # large enough offset
        boxes_offset = boxes_xyxy + cls_k.unsqueeze(1) * max_wh
        keep = nms(boxes_offset, conf_k, iou_thres)

        det = torch.cat(
            [boxes_xyxy[keep], conf_k[keep].unsqueeze(1), cls_k[keep].unsqueeze(1)],
            dim=1,
        )  # (N, 6)
        results.append(det)

    return results


def _xywh2xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert (cx, cy, w, h) to (x1, y1, x2, y2)."""
    out = boxes.clone()
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
    return out


def _xyxy_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute pairwise IoU between two sets of xyxy boxes."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    union = area1[:, None] + area2[None, :] - inter + 1e-7
    return inter / union


# ---------------------------------------------------------------------------
# TP matching
# ---------------------------------------------------------------------------


def match_predictions(
    pred_boxes: torch.Tensor,  # (N, 6): x1,y1,x2,y2,conf,cls
    gt_boxes: torch.Tensor,  # (M, 5): cls,cx,cy,w,h  normalised
    img_h: int,
    img_w: int,
    iou_thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Match predictions to ground-truth and compute TP flags.

    Returns:
        tp:       (N, len(iou_thresholds)) bool array
        conf:     (N,) confidence scores
        pred_cls: (N,) predicted class indices
    """
    N = len(pred_boxes)
    num_iou = len(iou_thresholds)

    tp = np.zeros((N, num_iou), dtype=bool)
    conf_arr = np.zeros(N)
    pred_cls_arr = np.zeros(N, dtype=int)

    if N == 0:
        return tp, conf_arr, pred_cls_arr

    # Convert GT from normalised xywh to pixel xyxy
    if len(gt_boxes) > 0:
        gt_xyxy = _gt_to_xyxy(gt_boxes, img_h, img_w)  # (M, 4)
        gt_cls = gt_boxes[:, 0].long()  # (M,)
    else:
        gt_xyxy = torch.zeros((0, 4))
        gt_cls = torch.zeros(0, dtype=torch.long)

    # Sort predictions by confidence (descending)
    order = pred_boxes[:, 4].argsort(descending=True)
    pred_boxes = pred_boxes[order]

    conf_arr = pred_boxes[:, 4].cpu().numpy()
    pred_cls_arr = pred_boxes[:, 5].cpu().numpy().astype(int)

    if len(gt_xyxy) == 0:
        return tp, conf_arr, pred_cls_arr

    iou_mat = _xyxy_iou(pred_boxes[:, :4], gt_xyxy.to(pred_boxes.device))  # (N, M)

    for thr_idx, iou_thr in enumerate(iou_thresholds):
        matched_gt = torch.full((len(gt_xyxy),), False, dtype=torch.bool)
        for i in range(N):
            if pred_cls_arr[i] < 0:
                continue
            # Filter by class match
            cls_mask = (gt_cls == pred_cls_arr[i]).to(pred_boxes.device)
            iou_row = iou_mat[i].clone()
            iou_row[~cls_mask] = -1.0
            iou_row[matched_gt] = -1.0

            best_iou, best_j = iou_row.max(0)
            if best_iou >= iou_thr:
                tp[i, thr_idx] = True
                matched_gt[best_j] = True

    return tp, conf_arr, pred_cls_arr


def _gt_to_xyxy(gt: torch.Tensor, img_h: int, img_w: int) -> torch.Tensor:
    """Convert GT labels (cls,cx,cy,w,h normalised) to pixel xyxy."""
    cx = gt[:, 1] * img_w
    cy = gt[:, 2] * img_h
    bw = gt[:, 3] * img_w
    bh = gt[:, 4] * img_h
    return torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dim=1)


# ---------------------------------------------------------------------------
# Tiny-object AP sub-metrics (VisDrone benchmark)
# ---------------------------------------------------------------------------


_TINY_AREA_THRESHOLDS = {
    "tiny1": (0, 32 * 32),
    "tiny2": (32 * 32, 64 * 64),
    "tiny3": (64 * 64, 128 * 128),
    "all": (0, float("inf")),
}


def _gt_area_px(gt_boxes: torch.Tensor, img_h: int, img_w: int) -> torch.Tensor:
    """Return pixel area for each GT box (cls,cx,cy,w,h normalised)."""
    bw = gt_boxes[:, 3] * img_w
    bh = gt_boxes[:, 4] * img_h
    return bw * bh


# ---------------------------------------------------------------------------
# AP computation (module-level so it is independently testable)
# ---------------------------------------------------------------------------


def _compute_ap(
    tp_list: list,
    conf_list: list,
    pred_cls_list: list,
    gt_cls_list: list,
    iou_thresholds: np.ndarray,
) -> tuple[float, float, float, float, float]:
    """
    Concatenate per-image stats and compute AP via ultralytics ap_per_class.

    Args:
        tp_list:       List of (N_i, T) bool arrays.
        conf_list:     List of (N_i,) float arrays.
        pred_cls_list: List of (N_i,) int arrays.
        gt_cls_list:   List of (M_i,) int arrays.
        iou_thresholds: Array of IoU thresholds used to build tp arrays.

    Returns:
        (precision, recall, f1, mAP50, mAP50-95) — all floats.
    """
    tp_all = (
        np.concatenate(tp_list, axis=0)
        if tp_list
        else np.zeros((0, len(iou_thresholds)))
    )
    conf_all = np.concatenate(conf_list) if conf_list else np.zeros(0)
    pred_cls_all = (
        np.concatenate(pred_cls_list) if pred_cls_list else np.zeros(0, dtype=int)
    )
    gt_cls_all = np.concatenate(gt_cls_list) if gt_cls_list else np.zeros(0, dtype=int)

    if len(pred_cls_all) == 0 or len(gt_cls_all) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    from ultralytics.utils.metrics import ap_per_class

    results = ap_per_class(tp_all, conf_all, pred_cls_all, gt_cls_all)
    # results: (tp, fp, precision, recall, f1, ap, unique_classes, ...)
    precision = results[2].mean()
    recall = results[3].mean()
    f1 = results[4].mean()
    ap50 = results[5][:, 0].mean()  # AP at IoU=0.50
    ap5095 = results[5].mean()  # AP averaged over IoU thresholds
    return float(precision), float(recall), float(f1), float(ap50), float(ap5095)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    nc: int,
    names: dict[int, str],
    model_type: str = "t2",
    conf_thres: float = 0.001,
    iou_thres: float = 0.65,
    iou_eval_thresholds: Optional[np.ndarray] = None,
    img_size: int = 640,
) -> dict:
    """
    Run full evaluation pass.

    Returns a dict with keys:
        mAP50, mAP50-95, precision, recall, F1
        mAP50_tiny1, mAP50_tiny2, mAP50_tiny3, mAP50_all
    """
    if iou_eval_thresholds is None:
        iou_eval_thresholds = np.linspace(0.5, 0.95, 10)

    model.eval()
    model.to(device)

    # Accumulate per-image stats
    all_tp: list[np.ndarray] = []
    all_conf: list[np.ndarray] = []
    all_pred_cls: list[np.ndarray] = []
    all_gt_cls: list[np.ndarray] = []

    # Tiny-object accumulators (keyed by size bin)
    tiny_tp: dict[str, list[np.ndarray]] = {k: [] for k in _TINY_AREA_THRESHOLDS}
    tiny_conf: dict[str, list[np.ndarray]] = {k: [] for k in _TINY_AREA_THRESHOLDS}
    tiny_pred_cls: dict[str, list[np.ndarray]] = {k: [] for k in _TINY_AREA_THRESHOLDS}
    tiny_gt_cls: dict[str, list[np.ndarray]] = {k: [] for k in _TINY_AREA_THRESHOLDS}

    for batch in loader:
        x_app = batch["X_app"].to(device)  # (B, 3, H, W)
        labels = batch["labels"]  # (N_total, 6): [batch_idx, cls, cx, cy, w, h]
        B = x_app.shape[0]
        H, W = x_app.shape[2], x_app.shape[3]

        if model_type == "t2":
            x_mot = batch["X_mot"].to(device)
            raw = model.predict(x_app, x_mot)
        else:
            raw = model(x_app)

        # raw is (B, 4+nc, A) or tuple thereof
        if isinstance(raw, (list, tuple)):
            raw = raw[0]  # take the decoded tensor

        preds = decode_predictions(
            raw, conf_thres=conf_thres, iou_thres=iou_thres, nc=nc
        )

        for i in range(B):
            # Extract GT for this image
            img_mask = labels[:, 0] == i
            gt_i = labels[img_mask, 1:]  # (M, 5): cls,cx,cy,w,h

            pred_i = preds[i]  # (N, 6) on device

            tp_i, conf_i, pred_cls_i = match_predictions(
                pred_i, gt_i, H, W, iou_eval_thresholds
            )
            gt_cls_i = (
                gt_i[:, 0].cpu().numpy().astype(int)
                if len(gt_i) > 0
                else np.zeros(0, dtype=int)
            )

            all_tp.append(tp_i)
            all_conf.append(conf_i)
            all_pred_cls.append(pred_cls_i)
            all_gt_cls.append(gt_cls_i)

            # --- Tiny sub-metrics ---
            if len(gt_i) > 0:
                areas = _gt_area_px(gt_i, H, W).cpu().numpy()
            else:
                areas = np.zeros(0)

            for bin_name, (a_min, a_max) in _TINY_AREA_THRESHOLDS.items():
                gt_mask = (areas >= a_min) & (areas < a_max)
                gt_bin = gt_i[torch.from_numpy(gt_mask)] if len(gt_i) > 0 else gt_i

                tp_b, conf_b, pred_cls_b = match_predictions(
                    pred_i, gt_bin, H, W, iou_eval_thresholds
                )
                gt_cls_b = (
                    gt_bin[:, 0].cpu().numpy().astype(int)
                    if len(gt_bin) > 0
                    else np.zeros(0, dtype=int)
                )

                tiny_tp[bin_name].append(tp_b)
                tiny_conf[bin_name].append(conf_b)
                tiny_pred_cls[bin_name].append(pred_cls_b)
                tiny_gt_cls[bin_name].append(gt_cls_b)

    # --- Compute AP ---
    p, r, f1, map50, map5095 = _compute_ap(
        all_tp, all_conf, all_pred_cls, all_gt_cls, iou_eval_thresholds
    )

    results = {
        "precision": p,
        "recall": r,
        "F1": f1,
        "mAP50": map50,
        "mAP50-95": map5095,
    }

    for bin_name in _TINY_AREA_THRESHOLDS:
        _, _, _, ap50_b, _ = _compute_ap(
            tiny_tp[bin_name],
            tiny_conf[bin_name],
            tiny_pred_cls[bin_name],
            tiny_gt_cls[bin_name],
            iou_eval_thresholds,
        )
        results[f"mAP50_{bin_name}"] = ap50_b

    return results
