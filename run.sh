#!/usr/bin/env bash
# Run the basketball tracker inside the 'basketball' conda environment.
#
# Speed guide (CPU, ~3-4x faster on Apple Silicon native):
#   yolov8n  skip=2 → ~15 min  (default, recommended)
#   yolov8m  skip=2 → ~60 min  (more accurate persons)
#   yolov8n  skip=3 → ~10 min  (fastest, slight quality loss)
#
# Usage:
#   ./run.sh
#   ./run.sh --model yolov8m.pt --skip 1   # max quality, very slow
#   ./run.sh --skip 3                       # fastest

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

conda run -n basketball python tracker.py \
    --cam1   videos/cam1.mp4 \
    --cam2   videos/cam2.mp4 \
    --output output.mp4 \
    --model  yolov8n.pt \
    --skip   2 \
    "$@"
