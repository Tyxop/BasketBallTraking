# Basketball Dual-Camera Tracker

Analyzes two simultaneous basketball game camera feeds to detect the ball, track players, and automatically select the most relevant camera per frame.

## How it works

Each frame from both cameras is scored independently:

| Event | Points |
|---|---|
| Ball visible in frame | +200 |
| Player touching / holding the ball | +300 |
| Per player in movement (on court) | +30 |
| Per player sitting / on bench | +5 |

The camera with the higher score is marked as **active** (green border). A single side-by-side output video is generated showing both feeds with annotations.

### Ball detection

Two complementary methods run on every frame:

1. **YOLOv8** — COCO class 32 (`sports ball`), confidence ≥ 0.25
2. **HSV color filter** — targets the orange range of a basketball; filters by area and circularity to reject jerseys and other orange regions

If YOLO misses the ball, the color detector acts as fallback (in practice it catches the ball reliably in most lighting conditions).

### Player classification

- **Moving** — DeepSORT track velocity ≥ 3.5 px/frame averaged over the last 15 frames
- **Sitting / bench** — low velocity *or* bounding box aspect ratio consistent with a seated person

Player IDs are stable across frames thanks to DeepSORT.

## Output

`output.mp4` — 1920×540 side-by-side annotated video:

```
┌─────────────────────┬─────────────────────┐
│  CAM 1  ◀ ACTIVE    │  CAM 2              │
│  Score: 560         │  Score: 245         │
│  Ball: YES  Mov: 5  │  Ball: no   Mov: 4  │
│                     │                     │
│  [M#3 +BALL] ──────►│                     │
│  [M#1] [M#2]        │  [S#7] [S#8]        │
└─────────────────────┴─────────────────────┘
```

## Architecture

The pipeline is split into two independent phases so analysis can run in parallel across multiple machines:

```
PHASE 1 — Analysis (ML heavy, run in parallel)
──────────────────────────────────────────────
Machine A  →  analyze.py --cam cam1.mp4  →  cam1.json
Machine B  →  analyze.py --cam cam2.mp4  →  cam2.json

PHASE 2 — Render (no ML, fast)
───────────────────────────────────────────────────────
Any machine  →  render.py --json1 cam1.json --json2 cam2.json  →  output.mp4
```

`analyze.py` automatically selects the best available device (CUDA → MPS → CPU).

### Speed by device

| Device | Model | skip | Time per camera (15 min game) |
|---|---|---|---|
| RTX 4090 | yolov8m | 1 | **~8 min** |
| RTX 5090 Mobile | yolov8m | 1 | ~10 min |
| M4 / M5 (MPS) | yolov8m | 1 | ~20 min |
| CPU x86 | yolov8n | 2 | ~15 min |

With a 2-machine camera-split the total time is `max(cam1, cam2)` — both cameras finish in the same time as one.

## Setup

### Single machine (conda)

```bash
conda create -n basketball python=3.11 -y
conda activate basketball
pip install ultralytics deep_sort_realtime opencv-python tqdm "numpy<2" "setuptools<71"
```

### NVIDIA GPU (Windows / Linux)

Install PyTorch with CUDA **before** the rest:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install ultralytics deep_sort_realtime opencv-python tqdm "numpy<2" "setuptools<71"
```

### Apple Silicon (M4 / M5) — native MPS

Use an `osx-arm64` conda environment (not Rosetta) to get MPS acceleration:

```bash
CONDA_SUBDIR=osx-arm64 conda create -n basketball python=3.11 -y
conda activate basketball
pip install ultralytics deep_sort_realtime opencv-python tqdm "numpy<2" "setuptools<71"
```

> Model weights (`yolov8n.pt` / `yolov8m.pt`) are downloaded automatically on first run.

## Usage

### Single machine

```bash
# Default — yolov8n, skip=2 (~15 min on CPU)
./run.sh

./run.sh --model yolov8m.pt --skip 1   # max quality
./run.sh --skip 3                       # fastest
```

### Distributed — 2 machines (camera split)

Run both `analyze.py` calls simultaneously on different machines, then render anywhere once both finish.

```bash
# Machine A (e.g. RTX 4090) — cam1
python analyze.py --cam cam1.mp4 --output cam1.json --model yolov8m.pt --skip 1

# Machine B (e.g. RTX 5090 Mobile) — cam2, at the same time
python analyze.py --cam cam2.mp4 --output cam2.json --model yolov8m.pt --skip 1

# Any machine — after both finish, copy cam1.json + cam2.json here
python render.py --cam1 cam1.mp4 --cam2 cam2.mp4 --json1 cam1.json --json2 cam2.json
```

### Distributed — 4 machines (camera + segment split)

Split each camera video in half so all four machines work simultaneously.

```bash
# Machine A — cam1, first half (frames 0–11600)
python analyze.py --cam cam1.mp4 --output cam1_a.json --start 0 --end 11600

# Machine B — cam1, second half
python analyze.py --cam cam1.mp4 --output cam1_b.json --start 11600

# Machine C — cam2, first half
python analyze.py --cam cam2.mp4 --output cam2_a.json --start 0 --end 11600

# Machine D — cam2, second half
python analyze.py --cam cam2.mp4 --output cam2_b.json --start 11600

# Merge segments, then render
python merge_segments.py --inputs cam1_a.json cam1_b.json --output cam1.json
python merge_segments.py --inputs cam2_a.json cam2_b.json --output cam2.json
python render.py --cam1 cam1.mp4 --cam2 cam2.mp4 --json1 cam1.json --json2 cam2.json
```

Total time with 4 machines: **~4–5 minutes**.

### `--skip N`

Process every Nth frame. Output video plays at the correct speed (`fps / N`).

| `--skip` | Speed | Quality |
|---|---|---|
| 1 | baseline | full |
| 2 | ~2× faster | good (default) |
| 3 | ~3× faster | acceptable |

## Files

| File | Purpose |
|---|---|
| `analyze.py` | Phase 1 — single-camera YOLO+DeepSORT → JSON |
| `render.py` | Phase 2 — JSON × 2 + videos → annotated video |
| `merge_segments.py` | Merge partial JSONs from segment-split runs |
| `run.sh` | Single-machine convenience wrapper |
| `run_distributed.sh` | Distributed workflow reference |

## Requirements

- Python 3.11
- [ultralytics](https://github.com/ultralytics/ultralytics) (YOLOv8)
- [deep_sort_realtime](https://github.com/levan92/deep_sort_realtime)
- OpenCV, NumPy, tqdm

GPU is optional but strongly recommended — CUDA gives ~17× speedup over CPU.
