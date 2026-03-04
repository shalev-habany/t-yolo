"""
data/visdrone_converter.py

Converts VisDrone dataset annotations to YOLO format and organises
video frames into named sequence folders for the temporal dataset loader.

Supported source layouts
------------------------
DET layout  (VisDrone2019-DET-train / -val / -test-dev)
  <split>/
    images/        *.jpg  (flat, one file per frame)
    annotations/   *.txt  (one file per image, 8 fields)
    Annotation format per line:
      x1,y1,w,h,score,category,truncation,occlusion

VID layout  (VisDrone2019-VID-val / VisDrone-VID-val / etc.)
  <split>/
    sequences/     <seq_name>/  *.jpg   (frames numbered 0000001.jpg …)
    annotations/   <seq_name>.txt       (one file per sequence, 10 fields)
    Annotation format per line:
      frame_index,target_id,x1,y1,w,h,score,category,truncation,occlusion

Object category mapping (VisDrone → 0-indexed):
  1:pedestrian 2:people 3:bicycle 4:car 5:van 6:truck
  7:tricycle 8:awning-tricycle 9:bus 10:motor 0:ignored-region 11:others

We map categories 1-10 to indices 0-9 and drop 0 (ignored) and 11 (others).

Output structure
----------------
  <dst>/
    sequences/
      train/  val/  test/
        <seq_name>/
          frames/   *.jpg
          labels/   *.txt  (YOLO cx cy w h, normalised)

Auto-detection
--------------
The converter probes <src> for known split folder names (DET and VID
variants).  Only the splits that are actually present are processed, so
passing a directory that contains only val data works fine.

Usage:
  python data/visdrone_converter.py --src /path/to/raw --dst ./data/VisDrone
"""

import argparse
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2


# VisDrone category id -> zero-indexed class id (drop 0=ignored, 11=others)
CAT_MAP = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9}

# Candidate split folder names -> canonical output split name.
# Both DET and VID naming variants are listed so the converter works
# regardless of which subset the user downloaded.
SPLIT_CANDIDATES: list[tuple[str, str]] = [
    # DET
    ("VisDrone2019-DET-train", "train"),
    ("VisDrone-DET-train", "train"),
    ("VisDrone2019-DET-val", "val"),
    ("VisDrone-DET-val", "val"),
    ("VisDrone2019-DET-test-dev", "test"),
    ("VisDrone-DET-test-dev", "test"),
    # VID
    ("VisDrone2019-VID-train", "train"),
    ("VisDrone-VID-train", "train"),
    ("VisDrone2019-VID-val", "val"),
    ("VisDrone-VID-val", "val"),
    ("VisDrone2019-VID-test-dev", "test"),
    ("VisDrone-VID-test-dev", "test"),
]


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------


def _bbox_to_yolo(
    x1: int, y1: int, bw: int, bh: int, img_w: int, img_h: int
) -> tuple[float, float, float, float]:
    cx = (x1 + bw / 2) / img_w
    cy = (y1 + bh / 2) / img_h
    nw = bw / img_w
    nh = bh / img_h
    return (
        max(0.0, min(1.0, cx)),
        max(0.0, min(1.0, cy)),
        max(0.0, min(1.0, nw)),
        max(0.0, min(1.0, nh)),
    )


def _img_size(img_path: Path) -> tuple[int, int] | None:
    """Return (width, height) or None if the image cannot be read."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    h, w = img.shape[:2]
    return w, h


# ---------------------------------------------------------------------------
# DET layout
# ---------------------------------------------------------------------------


def _convert_det_annotation(ann_path: Path, img_w: int, img_h: int) -> list[str]:
    """
    Parse a DET per-image annotation file (8 fields) and return YOLO lines.
      x1,y1,w,h,score,category,truncation,occlusion
    """
    lines = []
    for raw in ann_path.read_text().strip().splitlines():
        parts = raw.strip().split(",")
        if len(parts) < 6:
            continue
        x1, y1, bw, bh = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
        cat_id = int(parts[5])
        if cat_id not in CAT_MAP or bw <= 0 or bh <= 0:
            continue
        cx, cy, nw, nh = _bbox_to_yolo(x1, y1, bw, bh, img_w, img_h)
        lines.append(f"{CAT_MAP[cat_id]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return lines


def convert_det_split(src_split_dir: Path, dst_seq_dir: Path) -> None:
    """
    Convert a DET split.

    Images live flat in <split>/images/.
    Annotations are per-image files in <split>/annotations/.
    Images are grouped into sequences by their filename prefix
    (the part before the first '_').
    """
    images_dir = src_split_dir / "images"
    ann_dir = src_split_dir / "annotations"

    if not images_dir.exists():
        print(f"  [skip] images dir not found: {images_dir}")
        return

    dst_seq_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    if not image_files:
        print(f"  [skip] no images found in {images_dir}")
        return

    # Group by sequence prefix
    sequences: dict[str, list[Path]] = defaultdict(list)
    for img_path in image_files:
        seq_id = img_path.stem.split("_")[0] if "_" in img_path.stem else img_path.stem
        sequences[seq_id].append(img_path)

    total_imgs = total_labels = 0

    for seq_id, img_paths in sorted(sequences.items()):
        frames_dir = dst_seq_dir / seq_id / "frames"
        labels_dir = dst_seq_dir / seq_id / "labels"
        frames_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        for img_path in sorted(img_paths):
            dst_img = frames_dir / img_path.name
            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)

            ann_path = ann_dir / (img_path.stem + ".txt")
            if ann_path.exists():
                size = _img_size(img_path)
                if size is None:
                    continue
                w, h = size
                yolo_lines = _convert_det_annotation(ann_path, w, h)
                (labels_dir / (img_path.stem + ".txt")).write_text(
                    "\n".join(yolo_lines)
                )
                total_labels += 1

            total_imgs += 1

    print(
        f"  {dst_seq_dir.name}: {len(sequences)} sequences, "
        f"{total_imgs} images, {total_labels} label files"
    )


# ---------------------------------------------------------------------------
# VID layout
# ---------------------------------------------------------------------------


def _parse_vid_annotations(ann_path: Path) -> dict[int, list[str]]:
    """
    Parse a VID per-sequence annotation file (10 fields).
      frame_index,target_id,x1,y1,w,h,score,category,truncation,occlusion

    Returns {frame_index: [yolo_line, ...]} — sizes resolved later.
    We store raw bbox tuples here and resolve normalisation per-frame.
    """
    # Returns {frame_index: [(x1,y1,bw,bh,cat_id), ...]}
    raw: dict[int, list[tuple[int, int, int, int, int]]] = defaultdict(list)
    for line in ann_path.read_text().strip().splitlines():
        parts = line.strip().split(",")
        if len(parts) < 8:
            continue
        frame_idx = int(parts[0])
        x1, y1, bw, bh = int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
        cat_id = int(parts[7])
        if cat_id not in CAT_MAP or bw <= 0 or bh <= 0:
            continue
        raw[frame_idx].append((x1, y1, bw, bh, cat_id))
    return raw  # type: ignore[return-value]


def convert_vid_split(src_split_dir: Path, dst_seq_dir: Path) -> None:
    """
    Convert a VID split.

    Frames live in <split>/sequences/<seq_name>/*.jpg.
    Annotations are per-sequence files in <split>/annotations/<seq_name>.txt.
    """
    sequences_src = src_split_dir / "sequences"
    ann_dir = src_split_dir / "annotations"

    if not sequences_src.exists():
        print(f"  [skip] sequences dir not found: {sequences_src}")
        return

    seq_dirs = sorted(p for p in sequences_src.iterdir() if p.is_dir())
    if not seq_dirs:
        print(f"  [skip] no sequence folders in {sequences_src}")
        return

    dst_seq_dir.mkdir(parents=True, exist_ok=True)

    total_imgs = total_labels = 0

    for seq_dir in seq_dirs:
        seq_name = seq_dir.name
        frames_dst = dst_seq_dir / seq_name / "frames"
        labels_dst = dst_seq_dir / seq_name / "labels"
        frames_dst.mkdir(parents=True, exist_ok=True)
        labels_dst.mkdir(parents=True, exist_ok=True)

        # Load annotation index for this sequence (1-indexed frame numbers)
        ann_path = ann_dir / (seq_name + ".txt")
        ann_index: dict[int, list[tuple[int, int, int, int, int]]] = {}
        if ann_path.exists():
            ann_index = _parse_vid_annotations(ann_path)  # type: ignore[assignment]

        img_files = sorted(list(seq_dir.glob("*.jpg")) + list(seq_dir.glob("*.png")))

        for img_path in img_files:
            dst_img = frames_dst / img_path.name
            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)

            # VID frames are named 0000001.jpg; strip leading zeros for index
            try:
                frame_idx = int(img_path.stem)
            except ValueError:
                frame_idx = -1

            if frame_idx in ann_index:
                size = _img_size(img_path)
                if size is None:
                    total_imgs += 1
                    continue
                img_w, img_h = size
                yolo_lines = []
                for x1, y1, bw, bh, cat_id in ann_index[frame_idx]:
                    cx, cy, nw, nh = _bbox_to_yolo(x1, y1, bw, bh, img_w, img_h)
                    yolo_lines.append(
                        f"{CAT_MAP[cat_id]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"
                    )
                (labels_dst / (img_path.stem + ".txt")).write_text(
                    "\n".join(yolo_lines)
                )
                total_labels += 1

            total_imgs += 1

    print(
        f"  {dst_seq_dir.name}: {len(seq_dirs)} sequences, "
        f"{total_imgs} images, {total_labels} label files"
    )


# ---------------------------------------------------------------------------
# Layout detection
# ---------------------------------------------------------------------------


def _detect_layout(split_dir: Path) -> str | None:
    """
    Return 'det', 'vid', or None if the directory is not a recognised layout.
    """
    if (split_dir / "images").exists():
        return "det"
    if (split_dir / "sequences").exists():
        return "vid"
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert VisDrone (DET or VID) to YOLO temporal format"
    )
    parser.add_argument(
        "--src",
        required=True,
        help=(
            "Source directory.  Can be the dataset root (contains split "
            "sub-folders) OR a single split folder (e.g. VisDrone2019-VID-val)."
        ),
    )
    parser.add_argument(
        "--dst",
        default="./data/VisDrone",
        help="Output directory (default: ./data/VisDrone)",
    )
    args = parser.parse_args()

    src_root = Path(args.src).expanduser().resolve()
    dst_root = Path(args.dst).expanduser().resolve()

    if not src_root.exists():
        print(f"ERROR: source not found: {src_root}", file=sys.stderr)
        sys.exit(1)

    print(f"Converting VisDrone: {src_root} -> {dst_root}\n")

    converted = 0

    # --- Case 1: src itself is a single split folder ---
    layout = _detect_layout(src_root)
    if layout is not None:
        # Guess output split name from directory name
        folder = src_root.name.lower()
        if "train" in folder:
            dst_split = "train"
        elif "val" in folder:
            dst_split = "val"
        elif "test" in folder:
            dst_split = "test"
        else:
            dst_split = src_root.name  # fallback: keep folder name

        dst_seq_dir = dst_root / "sequences" / dst_split
        print(f"Detected single-split source ({layout.upper()} layout)")
        print(f"  {src_root.name} -> {dst_seq_dir}\n")

        if layout == "det":
            convert_det_split(src_root, dst_seq_dir)
        else:
            convert_vid_split(src_root, dst_seq_dir)

        converted += 1

    else:
        # --- Case 2: src is a root containing multiple split sub-folders ---
        seen_dst: set[str] = set()  # avoid processing the same output split twice

        for candidate, dst_split in SPLIT_CANDIDATES:
            if dst_split in seen_dst:
                continue
            split_dir = src_root / candidate
            if not split_dir.exists():
                continue

            layout = _detect_layout(split_dir)
            if layout is None:
                print(f"  [skip] unrecognised layout in {split_dir}")
                continue

            dst_seq_dir = dst_root / "sequences" / dst_split
            print(f"Processing {candidate} ({layout.upper()}) -> {dst_seq_dir}")

            if layout == "det":
                convert_det_split(split_dir, dst_seq_dir)
            else:
                convert_vid_split(split_dir, dst_seq_dir)

            seen_dst.add(dst_split)
            converted += 1

    if converted == 0:
        print(
            "ERROR: no recognised VisDrone split folders found in "
            f"{src_root}\n"
            "Expected one of:\n"
            + "\n".join(f"  {c}" for c, _ in SPLIT_CANDIDATES)
            + "\nor pass the split folder directly as --src.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\nDone ({converted} split(s) converted).")
    print(f"Dataset ready at: {dst_root}")
    print("Update configs/visdrone.yaml if you used a non-default --dst path.")


if __name__ == "__main__":
    main()
