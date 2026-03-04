"""
models/t2_yolov8.py

T2-YOLOv8 — Two-stream temporal detection model.

Paper: "Exploiting Temporal Context for Tiny Object Detection"
       Corsel et al., WACVW 2023

Architecture (Section 3.2 — T2-YOLOv5, ported to YOLOv8):

    Appearance stream:
        Input:  X_app ∈ R^{H×W×3}  — 3-channel grayscale temporal stack
        Backbone: YOLOv8{scale} (any size n/s/m/l/x)

    Motion stream:
        Input:  X_mot ∈ R^{H×W×2}  — 2-channel absolute frame difference
                channels: [|f_{t-s}-f_t|, |f_t-f_{t+s}|]
        Backbone: YOLOv8s (always small, per paper §3.2)
                  Exception: when main stream is Nano → motion stream is Nano

    Fusion:
        At each of the 3 detection scales (P3/8, P4/16, P5/32), concatenate
        the motion backbone output with the appearance backbone output BEFORE
        the neck processes that scale.

    Neck + Head:
        Standard YOLOv8 FPN neck sized to the fused channel widths.
        3 detection heads (P3, P4, P5) — no P2 or P6.

Training strategy (paper §3.2):
    - Transfer main stream weights from a trained T-YOLOv8 checkpoint.
    - Initialise motion stream from official YOLOv8s COCO weights.

Usage:
    model = T2YOLOv8(scale='x')
    model.load_pretrained(
        app_weights='runs/t_yolov8x_visdrone/best.pt',
        mot_weights='yolov8s.pt'      # auto-downloaded if not present
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from ultralytics.nn.modules import C2f, Conv, SPPF
from ultralytics.nn.modules.conv import Concat
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.torch_utils import initialize_weights

_HERE = Path(__file__).parent
_CONFIG_DIR = _HERE.parent / "configs"
_T_YAML = _CONFIG_DIR / "t_yolov8.yaml"

# Motion stream scale: same as main for Nano, else always Small.
_MOTION_SCALE: dict[str, str] = {
    "n": "n",
    "s": "s",
    "m": "s",
    "l": "s",
    "x": "s",
}


# ---------------------------------------------------------------------------
# Backbone extractor
# ---------------------------------------------------------------------------


class _BackboneExtractor(nn.Module):
    """
    Wraps the first 10 layers (backbone, layers 0-9) of a DetectionModel
    and returns feature maps at P3 (layer 4), P4 (layer 6), P5 (layer 9).

    The layer indices are fixed for all YOLOv8 scales because only the channel
    widths change with scale, not the graph topology.
    """

    _P3_IDX = 4
    _P4_IDX = 6
    _P5_IDX = 9

    def __init__(self, full_model: DetectionModel):
        super().__init__()
        # Copy the backbone layers (0-9)
        self.layers = nn.ModuleList(list(full_model.model.children())[:10])

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (B, C, H, W).

        Returns:
            (p3, p4, p5): Feature maps at strides 8, 16, 32.
        """
        prev = x
        saved: dict[int, torch.Tensor] = {}

        for i, layer in enumerate(self.layers):
            # All backbone layers have from=-1 (sequential), no skip connections
            prev = layer(prev)
            saved[i] = prev

        return saved[self._P3_IDX], saved[self._P4_IDX], saved[self._P5_IDX]


# ---------------------------------------------------------------------------
# FPN Neck (sized for fused channels)
# ---------------------------------------------------------------------------


class _FPNNeck(nn.Module):
    """
    YOLOv8-style FPN neck that accepts fused (concatenated) P3/P4/P5 features
    and returns three refined feature maps for the Detect head.

    Channel sizes are specified explicitly so the neck works with any fusion
    width (which varies by main-stream scale).

    Topology (identical to standard YOLOv8 neck):
        SPPF_out → Upsample → Concat(P4_fused) → C2f → [c_p4_neck]
                 → Upsample → Concat(P3_fused) → C2f → [c_p3_out]  (P3 output)
        c_p3_out → Conv → Concat(c_p4_neck)    → C2f → [c_p4_out]  (P4 output)
        c_p4_out → Conv → Concat(SPPF_out)     → C2f → [c_p5_out]  (P5 output)
    """

    def __init__(
        self,
        c_p3: int,  # fused P3 channels
        c_p4: int,  # fused P4 channels
        c_p5: int,  # fused P5 channels (also SPPF output channels)
    ):
        super().__init__()

        # --- Top-down path ---
        # SPPF output → upsample → concat with fused P4 → reduce
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        # After concat(SPPF_out, P4_fused): c_p5 + c_p4 channels
        c_td1 = c_p5 + c_p4
        self.c2f_td1 = C2f(c_td1, c_p4 // 2)  # produce ~half P4 channels

        # Reduce again → upsample → concat with fused P3 → reduce
        c_td2 = (c_p4 // 2) + c_p3
        self.c2f_td2 = C2f(c_td2, c_p3 // 2)  # P3 output

        # --- Bottom-up path ---
        c_p3_out = c_p3 // 2
        c_p4_neck = c_p4 // 2

        self.conv_bu1 = Conv(c_p3_out, c_p3_out, 3, 2)  # stride-2 conv
        c_bu1 = c_p3_out + c_p4_neck
        self.c2f_bu1 = C2f(c_bu1, c_p4_neck)  # P4 output

        self.conv_bu2 = Conv(c_p4_neck, c_p4_neck, 3, 2)
        c_bu2 = c_p4_neck + c_p5
        self.c2f_bu2 = C2f(c_bu2, c_p5 // 2)  # P5 output

        # Store output channel counts for downstream (Detect head)
        self.out_channels = (c_p3_out, c_p4_neck, c_p5 // 2)

    def forward(
        self,
        p3: torch.Tensor,
        p4: torch.Tensor,
        p5: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            p3, p4, p5: Fused backbone features at stride 8/16/32.

        Returns:
            (feat_p3, feat_p4, feat_p5): Neck output features for Detect head.
        """
        # Top-down
        td1 = self.c2f_td1(torch.cat([self.upsample(p5), p4], dim=1))
        feat_p3 = self.c2f_td2(torch.cat([self.upsample(td1), p3], dim=1))

        # Bottom-up
        feat_p4 = self.c2f_bu1(torch.cat([self.conv_bu1(feat_p3), td1], dim=1))
        feat_p5 = self.c2f_bu2(torch.cat([self.conv_bu2(feat_p4), p5], dim=1))

        return feat_p3, feat_p4, feat_p5


# ---------------------------------------------------------------------------
# T2YOLOv8 main model
# ---------------------------------------------------------------------------


class T2YOLOv8(nn.Module):
    """
    T2-YOLOv8 two-stream temporal detection model.

    Args:
        scale:   Main stream scale — one of 'n', 's', 'm', 'l', 'x'.
        nc:      Number of detection classes (10 for VisDrone).
        verbose: Print architecture info.
    """

    def __init__(
        self,
        scale: str = "x",
        nc: int = 10,
        verbose: bool = True,
    ):
        super().__init__()
        self.scale = scale
        self.nc = nc

        if not _T_YAML.exists():
            raise FileNotFoundError(f"Model config not found: {_T_YAML}")

        import yaml

        # --- Build full appearance-stream model (to extract backbone) ---
        with open(_T_YAML) as f:
            cfg_app = yaml.safe_load(f)
        cfg_app["scale"] = scale
        _full_app = DetectionModel(cfg_app, ch=3, nc=nc, verbose=False)

        # --- Build full motion-stream model (always small, ch=2) ---
        mot_scale = _MOTION_SCALE[scale]
        with open(_T_YAML) as f:
            cfg_mot = yaml.safe_load(f)
        cfg_mot["scale"] = mot_scale
        _full_mot = DetectionModel(cfg_mot, ch=2, nc=nc, verbose=False)

        # --- Wrap backbones ---
        self.app_backbone = _BackboneExtractor(_full_app)
        self.mot_backbone = _BackboneExtractor(_full_mot)

        # --- Compute fused channel sizes ---
        with torch.no_grad():
            dummy_app = torch.zeros(1, 3, 256, 256)
            dummy_mot = torch.zeros(1, 2, 256, 256)
            app_p3, app_p4, app_p5 = self.app_backbone(dummy_app)
            mot_p3, mot_p4, mot_p5 = self.mot_backbone(dummy_mot)

        c_p3 = app_p3.shape[1] + mot_p3.shape[1]
        c_p4 = app_p4.shape[1] + mot_p4.shape[1]
        c_p5 = app_p5.shape[1] + mot_p5.shape[1]

        if verbose:
            LOGGER.info(
                f"T2-YOLOv8{scale}: app backbone P3/P4/P5 = "
                f"{app_p3.shape[1]}/{app_p4.shape[1]}/{app_p5.shape[1]}"
            )
            LOGGER.info(
                f"T2-YOLOv8{scale}: mot backbone (YOLOv8{mot_scale}) P3/P4/P5 = "
                f"{mot_p3.shape[1]}/{mot_p4.shape[1]}/{mot_p5.shape[1]}"
            )
            LOGGER.info(
                f"T2-YOLOv8{scale}: fused channels P3/P4/P5 = {c_p3}/{c_p4}/{c_p5}"
            )

        # --- Neck ---
        self.neck = _FPNNeck(c_p3, c_p4, c_p5)
        c_det = self.neck.out_channels  # (c_p3_out, c_p4_out, c_p5_out)

        if verbose:
            LOGGER.info(
                f"T2-YOLOv8{scale}: Detect head input channels = "
                f"{c_det[0]}/{c_det[1]}/{c_det[2]}"
            )

        # --- Detect head ---
        self.detect = Detect(nc=nc, ch=c_det)

        # Compute strides by dry-run
        self._init_strides()

        # Initialise weights
        initialize_weights(self)
        self.detect.bias_init()

        # Attach default hyperparameters so v8DetectionLoss can initialise.
        # v8DetectionLoss expects model.args and model.model[-1] (Detect head).
        from ultralytics.utils import DEFAULT_CFG

        self.args = DEFAULT_CFG

        # Loss (initialised lazily in forward)
        self.criterion: Optional[v8DetectionLoss] = None

        if verbose:
            total = sum(p.numel() for p in self.parameters())
            LOGGER.info(f"T2-YOLOv8{scale}: {total:,} parameters")

    # ------------------------------------------------------------------
    # Compatibility shim for v8DetectionLoss
    # ------------------------------------------------------------------

    @property
    def model(self) -> list:
        """
        Expose self.detect as model[-1] so that v8DetectionLoss(self) works.
        v8DetectionLoss does: m = model.model[-1]
        """
        return [self.detect]

    # ------------------------------------------------------------------
    # Stride initialisation
    # ------------------------------------------------------------------

    def _init_strides(self) -> None:
        """Compute Detect head strides via a dry forward pass."""
        s = 256
        self.detect.training = True
        with torch.no_grad():
            dummy_app = torch.zeros(1, 3, s, s)
            dummy_mot = torch.zeros(1, 2, s, s)
            feats = self._extract_feats(dummy_app, dummy_mot)
            preds = self.detect(list(feats))

        # preds[1] contains the raw feature list when training=True
        if isinstance(preds, (list, tuple)) and isinstance(preds[0], torch.Tensor):
            # preds is the tuple from Detect (distribution, cls list)
            feat_list = feats
        else:
            feat_list = feats

        self.detect.stride = torch.tensor([s / f.shape[-2] for f in feat_list])
        self.stride = self.detect.stride

    # ------------------------------------------------------------------
    # Core forward helpers
    # ------------------------------------------------------------------

    def _extract_feats(
        self,
        x_app: torch.Tensor,
        x_mot: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract and fuse features from both streams.

        Returns (feat_p3, feat_p4, feat_p5): neck output features.
        """
        app_p3, app_p4, app_p5 = self.app_backbone(x_app)
        mot_p3, mot_p4, mot_p5 = self.mot_backbone(x_mot)

        # Fusion: concatenate motion features onto appearance features at each scale
        fused_p3 = torch.cat([app_p3, mot_p3], dim=1)
        fused_p4 = torch.cat([app_p4, mot_p4], dim=1)
        fused_p5 = torch.cat([app_p5, mot_p5], dim=1)

        return self.neck(fused_p3, fused_p4, fused_p5)

    # ------------------------------------------------------------------
    # Training / inference forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor | dict,
    ) -> torch.Tensor | tuple | dict:
        """
        Forward pass.

        Training:
            x is a dict with keys:
                'img'    — X_app tensor (B, 3, H, W)
                'X_mot'  — X_mot tensor (B, 2, H, W)
                'bboxes' — GT boxes
                'cls'    — GT classes
                (other standard ultralytics batch keys)
            Returns loss dict.

        Inference:
            x is a dict {'img': X_app, 'X_mot': X_mot} or
            a tuple (X_app, X_mot).
            Returns Detect head output.
        """
        if isinstance(x, dict):
            if "bboxes" in x or "cls" in x:
                # Training mode
                return self.loss(x)
            else:
                # Inference via dict
                x_app = x["img"]
                x_mot = x["X_mot"]
                feats = self._extract_feats(x_app, x_mot)
                return self.detect(list(feats))
        elif isinstance(x, (list, tuple)):
            x_app, x_mot = x[0], x[1]
            feats = self._extract_feats(x_app, x_mot)
            return self.detect(list(feats))
        else:
            raise TypeError(
                "T2YOLOv8.forward() expects a dict or (X_app, X_mot) tuple. "
                f"Got {type(x)}."
            )

    def predict(
        self,
        x_app: torch.Tensor,
        x_mot: torch.Tensor,
    ) -> torch.Tensor | tuple:
        """Run inference on a batch. Returns Detect head output."""
        feats = self._extract_feats(x_app, x_mot)
        return self.detect(list(feats))

    def loss(self, batch: dict, preds=None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute YOLOv8 detection loss.

        Args:
            batch: dict with 'img' (X_app), 'X_mot', and GT label keys.
            preds: Optional pre-computed predictions (skips forward pass).

        Returns:
            (total_loss, detached_loss_items)  — same convention as ultralytics.
        """
        if self.criterion is None:
            self.criterion = v8DetectionLoss(self)

        if preds is None:
            x_app = batch["img"]
            x_mot = batch["X_mot"]
            feats = self._extract_feats(x_app, x_mot)
            preds = self.detect(list(feats))

        return self.criterion(preds, batch)

    # ------------------------------------------------------------------
    # Pretrained weight loading
    # ------------------------------------------------------------------

    def load_pretrained(
        self,
        app_weights: Optional[str] = None,
        mot_weights: Optional[str] = None,
    ) -> "T2YOLOv8":
        """
        Load pretrained weights.

        app_weights: Path to a trained T-YOLOv8 checkpoint (main stream).
                     If None, skipped.
        mot_weights: Path to official YOLOv8s COCO weights (motion stream).
                     If None, uses 'yolov8{mot_scale}.pt' (auto-downloaded).

        Per paper §3.2: transfer T-YOLOv8 weights to main stream;
        initialise motion stream from YOLOv8s COCO weights.
        """
        mot_scale = _MOTION_SCALE[self.scale]

        # --- Load motion stream weights (YOLOv8s/n COCO) ---
        if mot_weights is None:
            mot_weights = f"yolov8{mot_scale}.pt"
        LOGGER.info(f"T2-YOLOv8: loading motion stream weights from {mot_weights}")
        _load_backbone_weights(self.mot_backbone, mot_weights)

        # --- Load appearance stream weights (trained T-YOLOv8) ---
        if app_weights is not None:
            LOGGER.info(
                f"T2-YOLOv8: loading appearance stream weights from {app_weights}"
            )
            _load_backbone_weights(self.app_backbone, app_weights)
        else:
            LOGGER.info(
                "T2-YOLOv8: no appearance stream weights provided; backbone is random"
            )

        return self

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        mot_scale = _MOTION_SCALE[self.scale]
        total = sum(p.numel() for p in self.parameters())
        return (
            f"T2YOLOv8(scale={self.scale!r}, mot_scale={mot_scale!r}, "
            f"nc={self.nc}, params={total:,})"
        )


# ---------------------------------------------------------------------------
# Helper: partial weight transfer into a _BackboneExtractor
# ---------------------------------------------------------------------------


def _load_backbone_weights(
    backbone: _BackboneExtractor,
    weights_path: str,
) -> None:
    """
    Transfer matching weights from a YOLOv8 checkpoint into a _BackboneExtractor.
    Keys that do not match in shape are skipped.
    """
    from ultralytics.nn.tasks import load_checkpoint

    src_model, _ = load_checkpoint(weights_path, device="cpu")  # (model, ckpt_dict)

    # The _BackboneExtractor stores layers as self.layers[0..9].
    # In the source DetectionModel they are model.model[0..9].
    # Build a key mapping: "layers.{i}.*" -> "model.{i}.*"
    state_src = src_model.state_dict()
    state_dst = backbone.state_dict()

    transfer: dict[str, torch.Tensor] = {}
    skipped = []

    # Map "model.{i}.XXX" -> "layers.{i}.XXX"  for i in 0..9
    for k_src, v_src in state_src.items():
        parts = k_src.split(".")
        if len(parts) < 2:
            continue
        # Source key starts with "model.{idx}."
        if parts[0] != "model":
            continue
        try:
            idx = int(parts[1])
        except ValueError:
            continue
        if idx > 9:
            continue
        k_dst = "layers." + ".".join(parts[1:])
        if k_dst in state_dst and state_dst[k_dst].shape == v_src.shape:
            transfer[k_dst] = v_src
        else:
            skipped.append(k_src)

    state_dst.update(transfer)
    backbone.load_state_dict(state_dst)

    LOGGER.info(
        f"  Transferred {len(transfer)} tensors "
        f"({len(skipped)} skipped due to shape mismatch)"
    )
