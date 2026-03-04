"""
data/visdrone_converter.py

Converts VisDrone dataset annotations to YOLO format and organises
video frames into named sequence folders for the temporal dataset loader.

VisDrone annotation format (per line):
  <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>

Object category mapping (VisDrone → 0-indexed):
  1:pedestrian 2:people 3:bicycle 4:car 5:van 6:truck
  7:tricycle 8:awning-tricycle 9:bus 10:motor 0:ignored-region 11:others

We map categories 1-10 to indices 0-9 and drop 0 (ignored) and 11 (others).

Output structure:
  data/VisDrone/
    sequences/
      train/
        sequence_0001/
          frames/          <- .jpg images
          labels/          <- .txt YOLO labels (same stem as frame)
      val/  ...
      test/ ...

Usage:
  python data/visdrone_converter.py \
      --src  /path/to/VisDroneDET2019/          \
      --dst  ./data/VisDrone
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


# VisDrone category id -> zero-indexed class id (drop 0=ignored, 11=others)
CAT_MAP = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9}

SPLITS = {
    "VisDrone2019-DET-train": "train",
    "VisDrone2019-DET-val": "val",
    "VisDrone2019-DET-test-dev": "test",
}


def convert_annotation(ann_path: Path, img_w: int, img_h: int) -> list[str]:
    """Convert one VisDrone .txt annotation file to YOLO format lines."""
    lines = []
    for raw in ann_path.read_text().strip().splitlines():
        if not raw.strip():
            continue
        parts = raw.strip().split(",")
        if len(parts) < 6:
            continue
        x1, y1, bw, bh = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
        cat_id = int(parts[5])

        if cat_id not in CAT_MAP:
            continue  # skip ignored-region and others
        if bw <= 0 or bh <= 0:
            continue

        cls = CAT_MAP[cat_id]

        # YOLO: cx, cy, w, h — all normalised
        cx = (x1 + bw / 2) / img_w
        cy = (y1 + bh / 2) / img_h
        nw = bw / img_w
        nh = bh / img_h

        # Clamp to [0, 1]
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        nw = max(0.0, min(1.0, nw))
        nh = max(0.0, min(1.0, nh))

        lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return lines


def convert_split(src_split_dir: Path, dst_seq_dir: Path) -> None:
    """Convert one dataset split (train / val / test)."""
    images_dir = src_split_dir / "images"
    ann_dir = src_split_dir / "annotations"

    if not images_dir.exists():
        print(f"  [skip] {images_dir} not found")
        return

    dst_seq_dir.mkdir(parents=True, exist_ok=True)

    # VisDrone DET stores each image as a standalone file (not sequences).
    # We group them by their filename prefix (sequence_XXXX_frameYYYYYY)
    # or treat each image as its own "sequence of 1" — the temporal loader
    # handles boundary duplication automatically.
    #
    # For VisDrone-VID (recommended for temporal models), images are already
    # in sequence folders. We support both layouts here.

    image_files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))

    # Group by sequence prefix (works for both DET and VID naming)
    from collections import defaultdict

    sequences: dict[str, list[Path]] = defaultdict(list)
    for img_path in image_files:
        stem = img_path.stem  # e.g. "0000001_00001_d_0000001"
        # VisDrone-VID: prefix is first 7 chars ("0000001")
        # VisDrone-DET: no natural grouping — use full name as single-frame seq
        seq_id = stem.split("_")[0] if "_" in stem else stem
        sequences[seq_id].append(img_path)

    total_imgs = 0
    total_labels = 0

    for seq_id, img_paths in sequences.items():
        seq_dir = dst_seq_dir / f"seq_{seq_id}"
        frames_dir = seq_dir / "frames"
        labels_dir = seq_dir / "labels"
        frames_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        for img_path in sorted(img_paths):
            # Copy / symlink image
            dst_img = frames_dir / img_path.name
            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)

            # Convert annotation
            ann_path = ann_dir / (img_path.stem + ".txt")
            if ann_path.exists():
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                h, w = img.shape[:2]
                yolo_lines = convert_annotation(ann_path, w, h)
                lbl_path = labels_dir / (img_path.stem + ".txt")
                lbl_path.write_text("\n".join(yolo_lines))
                total_labels += 1

            total_imgs += 1

    print(
        f"  {dst_seq_dir.name}: {len(sequences)} sequences, "
        f"{total_imgs} images, {total_labels} label files"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Convert VisDrone to YOLO temporal format"
    )
    parser.add_argument("--src", required=True, help="VisDrone dataset root directory")
    parser.add_argument("--dst", default="./data/VisDrone", help="Output directory")
    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)

    print(f"Converting VisDrone: {src_root} -> {dst_root}")

    for src_split_name, dst_split_name in SPLITS.items():
        src_split = src_root / src_split_name
        if not src_split.exists():
            # Also try without year suffix
            src_split = src_root / src_split_name.replace("2019-", "")
        if not src_split.exists():
            print(f"  [skip] {src_split_name} not found in {src_root}")
            continue

        dst_seq_dir = dst_root / "sequences" / dst_split_name
        print(f"\nProcessing {src_split_name} -> {dst_seq_dir}")
        convert_split(src_split, dst_seq_dir)

    print("\nDone. Update configs/visdrone.yaml path if needed.")


if __name__ == "__main__":
    main()
