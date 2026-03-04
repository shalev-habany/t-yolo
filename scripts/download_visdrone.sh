#!/usr/bin/env bash
# scripts/download_visdrone.sh
#
# Download VisDrone2019 DET dataset splits from the official mirrors.
#
# Usage:
#   bash scripts/download_visdrone.sh [--dst <dir>]
#
# The three splits (train / val / test-dev) are downloaded to:
#   <dst>/VisDrone2019-DET-train/
#   <dst>/VisDrone2019-DET-val/
#   <dst>/VisDrone2019-DET-test-dev/
#
# After downloading run:
#   python data/visdrone_converter.py --src <dst> --dst data/VisDrone
#
# NOTE: The official VisDrone download links require registration on the
# VisDrone workshop site.  If the URLs below have expired, download the
# archives manually and place them in <dst> before running this script.
#
# Official site: https://github.com/VisDrone/VisDrone-Dataset

set -euo pipefail

# ---------------------------------------------------------------------------
# Default destination
# ---------------------------------------------------------------------------
DST="./data/raw_visdrone"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dst)
            DST="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

mkdir -p "$DST"
echo "Downloading VisDrone2019-DET to: $DST"

# ---------------------------------------------------------------------------
# Download URLs
# These are the Google Drive IDs used by the official VisDrone repository.
# (Using parallel arrays for bash 3.2 compatibility on macOS)
# ---------------------------------------------------------------------------

ARCHIVE_NAMES=(
    "VisDrone2019-DET-train.zip"
    "VisDrone2019-DET-val.zip"
    "VisDrone2019-DET-test-dev.zip"
)

ARCHIVE_IDS=(
    "1a2oHjcEcwXP3DrptG-GQ2YGCV3-3n9bm"
    "1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59"
    "1PFdW_VFSCfZ_sTSZAGjQdifF_Xd5mf0V"
)

download_gdrive() {
    local file_id="$1"
    local out_path="$2"

    if command -v gdown &>/dev/null; then
        gdown "https://drive.google.com/uc?id=${file_id}" -O "$out_path"
    elif command -v wget &>/dev/null; then
        # wget with cookie handling for Google Drive
        wget --load-cookies /tmp/gdrive_cookies.txt \
             "https://docs.google.com/uc?export=download&confirm=$(
                wget --quiet --save-cookies /tmp/gdrive_cookies.txt \
                     --keep-session-cookies \
                     --no-check-certificate \
                     "https://docs.google.com/uc?export=download&id=${file_id}" \
                     -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p'
             )&id=${file_id}" \
             -O "$out_path" \
             --no-check-certificate
    else
        echo "ERROR: Neither 'gdown' nor 'wget' found. Install one of them:" >&2
        echo "  pip install gdown   OR   apt-get install wget" >&2
        exit 1
    fi
}

for i in "${!ARCHIVE_NAMES[@]}"; do
    archive="${ARCHIVE_NAMES[$i]}"
    file_id="${ARCHIVE_IDS[$i]}"
    out="$DST/$archive"
    split_name="${archive%.zip}"

    if [[ -d "$DST/$split_name" ]]; then
        echo "  [skip] $split_name already extracted"
        continue
    fi

    if [[ ! -f "$out" ]]; then
        echo "  Downloading $archive ..."
        download_gdrive "$file_id" "$out"
    else
        echo "  [skip] $archive already downloaded"
    fi

    echo "  Extracting $archive ..."
    unzip -q "$out" -d "$DST"
    rm -f "$out"
    echo "  Done: $DST/$split_name"
done

echo ""
echo "Download complete."
echo ""
echo "Next step — convert to YOLO temporal format:"
echo "  python data/visdrone_converter.py --src $DST --dst data/VisDrone"
