#!/usr/bin/env python3
"""
scripts/convert_visdrone.py

Convenience runner for data/visdrone_converter.py.

Downloads VisDrone (optional) and then converts the raw dataset to the
YOLO temporal format expected by TemporalDataset.

Usage:
    # If you already downloaded VisDrone to /data/raw_visdrone:
    python scripts/convert_visdrone.py --src /data/raw_visdrone

    # Override output directory:
    python scripts/convert_visdrone.py --src /data/raw_visdrone --dst ./data/VisDrone

After conversion the directory structure will be:
    data/VisDrone/
        sequences/
            train/  val/  test/
                seq_<id>/
                    frames/   *.jpg
                    labels/   *.txt  (YOLO format)

Then update configs/visdrone.yaml if you used a non-default --dst path.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).parent
_ROOT = _HERE.parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert VisDrone dataset to YOLO temporal format"
    )
    parser.add_argument(
        "--src",
        required=True,
        help="Path to raw VisDrone root (contains VisDrone2019-DET-train/ etc.)",
    )
    parser.add_argument(
        "--dst",
        default=str(_ROOT / "data" / "VisDrone"),
        help="Output directory for converted dataset (default: data/VisDrone)",
    )
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    if not src.exists():
        print(f"ERROR: Source directory not found: {src}", file=sys.stderr)
        print(
            "Download VisDrone first:\n  bash scripts/download_visdrone.sh --dst "
            f"{src}",
            file=sys.stderr,
        )
        sys.exit(1)

    converter = _ROOT / "data" / "visdrone_converter.py"
    if not converter.exists():
        print(f"ERROR: Converter not found: {converter}", file=sys.stderr)
        sys.exit(1)

    print(f"Source : {src}")
    print(f"Output : {dst}")
    print()

    cmd = [sys.executable, str(converter), "--src", str(src), "--dst", str(dst)]
    print(f"Running: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print("\nConversion failed.", file=sys.stderr)
        sys.exit(result.returncode)

    print("\nConversion complete.")
    print(f"\nDataset ready at: {dst}")
    print("\nMake sure configs/visdrone.yaml has:")
    print(f"  path: {dst}")


if __name__ == "__main__":
    main()
