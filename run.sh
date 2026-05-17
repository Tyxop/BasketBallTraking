#!/usr/bin/env bash
# Single-machine run (uses analyze.py + render.py internally).
#
# Speed guide (CPU vs GPU):
#   CPU (x86)       yolov8n  skip=2  →  ~15 min
#   M4/M5  (MPS)   yolov8m  skip=2  →  ~8 min   (needs arm64 conda, see README)
#   RTX 4090 (CUDA) yolov8m  skip=1  →  ~3 min
#
# Usage:
#   ./run.sh
#   ./run.sh --model yolov8m.pt --skip 1

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

MODEL=${MODEL:-yolov8n.pt}
SKIP=${SKIP:-2}

# Parse extra args
while [[ $# -gt 0 ]]; do
  case $1 in
    --model) MODEL=$2; shift 2;;
    --skip)  SKIP=$2;  shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

echo "=== PHASE 1: Analyzing cam1 ==="
conda run --no-capture-output -n basketball python -u analyze.py \
    --cam videos/cam1.mp4 --output cam1.json --model "$MODEL" --skip "$SKIP"

echo ""
echo "=== PHASE 1: Analyzing cam2 ==="
conda run --no-capture-output -n basketball python -u analyze.py \
    --cam videos/cam2.mp4 --output cam2.json --model "$MODEL" --skip "$SKIP"

echo ""
echo "=== PHASE 2: Rendering ==="
conda run --no-capture-output -n basketball python -u render.py \
    --cam1 videos/cam1.mp4 --cam2 videos/cam2.mp4 \
    --json1 cam1.json --json2 cam2.json \
    --output output.mp4
