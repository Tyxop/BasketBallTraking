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

`output.json` — per-run statistics (% time ball in each camera, % time each camera was active).

## Setup

```bash
conda create -n basketball python=3.11 -y
conda activate basketball
pip install ultralytics deep_sort_realtime opencv-python "numpy<2" "setuptools<71"
```

> Model weights (`yolov8n.pt` / `yolov8m.pt`) are downloaded automatically on first run.

## Usage

```bash
# Default — fast (yolov8n, process every 2nd frame, ~15 min for a 15-min game)
./run.sh

# Override any parameter
./run.sh --model yolov8m.pt --skip 1   # max quality, ~60 min
./run.sh --skip 3                       # fastest, ~10 min

# Manual
conda run -n basketball python tracker.py \
    --cam1   videos/cam1.mp4 \
    --cam2   videos/cam2.mp4 \
    --output output.mp4 \
    --model  yolov8n.pt \
    --skip   2
```

### `--skip N`

Process every Nth frame. Output video plays at the correct speed (`fps / N`). Recommended values:

| `--skip` | Speed (CPU) | Quality |
|---|---|---|
| 1 | slowest | full |
| 2 | ~2× faster | good (default) |
| 3 | ~3× faster | acceptable |

## Requirements

- Python 3.11
- [ultralytics](https://github.com/ultralytics/ultralytics) (YOLOv8)
- [deep_sort_realtime](https://github.com/levan92/deep_sort_realtime)
- OpenCV, NumPy

No GPU required — runs on CPU. Apple Silicon / CUDA automatically used if available.
