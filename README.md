# Basketball Dual-Camera Tracker

Analyzes two simultaneous basketball game camera feeds to detect the ball, track players, and select the most relevant camera per frame — either automatically or through a visual editor.

---

## Quick start

```bash
# 1. Create environment
conda create -n basketball python=3.11 -y
conda activate basketball
pip install ultralytics deep_sort_realtime opencv-python tqdm PyQt6 "numpy<2" "setuptools<71"

# 2. Open the editor
./gui.sh videos/cam1.mp4 videos/cam2.mp4
```

---

## GUI editor

`gui.py` is a dark-themed video editor for reviewing and cutting between two cameras.

```
┌─────────────────────────────────────────────────────────┐
│  CAM 1  (blue border = active)   CAM 2                  │
│  ┌─────────────────────┐  ┌─────────────────────┐       │
│  │                     │  │                     │       │
│  │      [video]        │  │      [video]        │       │
│  │                     │  │                     │       │
│  │      ◀ ACTIVA       │  │                     │       │
│  └─────────────────────┘  └─────────────────────┘       │
│                                                         │
│  [████████████████░░░░░░░░░████████████████████]        │
│   cam1 (blue)        cam2 (orange)    cam1              │
│  ────────────────────────▲──────────────────────        │
│                                                         │
│  ▶  ══════════○══════════  07:12 / 15:38                │
│                                                         │
│  [1] Corte→CAM1  [2] Corte→CAM2  [↩ Deshacer]          │
│                                                         │
│  Análisis automático:                                   │
│  [yolov8n ▼]  [skip 2 ▼]  [🔍 Analizar automáticamente]│
│  ✓ 12 cortes detectados. Revisa y ajusta si lo necesitas│
│                                                         │
│  [📂 Abrir]          [▶ Preview]    [🎬 Generar vídeo]  │
└─────────────────────────────────────────────────────────┘
```

### Recommended workflow

1. Open the editor: `./gui.sh` (or `./gui.sh videos/cam1.mp4 videos/cam2.mp4`)
2. Click **🔍 Analizar automáticamente** — YOLO+DeepSORT runs in the background and populates the timeline with automatic cuts
3. Press **▶ Preview** to play through and review the result
4. Fine-tune manually: scrub to any point and press `1` or `2` to override a cut
5. Click **🎬 Generar vídeo** to export the final MP4

### Keyboard shortcuts

| Key | Action |
|---|---|
| `Space` | Play / Pause |
| `1` | Mark cut → CAM 1 at current position |
| `2` | Mark cut → CAM 2 at current position |
| `Z` | Undo last cut |
| `←` / `→` | Step 1 frame |
| `Shift + ←/→` | Jump 30 frames |

### Auto-analyze options

| Option | Values | Notes |
|---|---|---|
| Model | yolov8n / yolov8s / yolov8m | n = fast, m = most accurate |
| Skip | 1 / 2 / 3 | Process every Nth frame |

The algorithm smooths per-frame scores over a 2-second window and applies a 3-second hysteresis filter so brief score fluctuations don't cause spurious cuts.

---

## How the scoring works

Each frame from both cameras is scored independently:

| Event | Points |
|---|---|
| Ball visible in frame | +200 |
| Player touching the ball | +300 |
| Per player in movement (on court) | +30 |
| Per player sitting / on bench | +5 |

The camera with the higher score wins that frame. A player with the ball outweighs any number of players without one.

### Ball detection

Two methods run on every frame — whichever fires first wins:

1. **YOLOv8** — COCO class 32 (`sports ball`), confidence ≥ 0.25
2. **HSV color filter** — targets the orange range of a basketball, filtered by area and circularity to reject jerseys

### Player classification

- **Moving** — DeepSORT track velocity ≥ 3.5 px/frame over the last 15 frames
- **Sitting / bench** — low velocity *or* bounding box width/height ratio consistent with a seated person

---

## Headless pipeline (CLI)

For batch processing without the GUI — useful for distributed runs across multiple machines.

### Architecture

```
PHASE 1 — Analysis (ML heavy, parallelizable)
─────────────────────────────────────────────
Machine A  →  analyze.py --cam cam1.mp4  →  cam1.json
Machine B  →  analyze.py --cam cam2.mp4  →  cam2.json

PHASE 2 — Render (no ML, fast)
──────────────────────────────
Any machine  →  render.py --json1 cam1.json --json2 cam2.json  →  output.mp4
```

`analyze.py` auto-detects the best device (CUDA → MPS → CPU).

### Single machine

```bash
./run.sh                            # yolov8n, skip=2 (~15 min on CPU)
./run.sh --model yolov8m.pt --skip 1   # max quality
```

### 2 machines — camera split

```bash
# Machine A (e.g. RTX 4090)
python analyze.py --cam cam1.mp4 --output cam1.json --model yolov8m.pt --skip 1

# Machine B (e.g. RTX 5090 Mobile) — at the same time
python analyze.py --cam cam2.mp4 --output cam2.json --model yolov8m.pt --skip 1

# Any machine — once both finish
python render.py --cam1 cam1.mp4 --cam2 cam2.mp4 --json1 cam1.json --json2 cam2.json
```

### 4 machines — camera + segment split

```bash
python analyze.py --cam cam1.mp4 --output cam1_a.json --start 0     --end 11600
python analyze.py --cam cam1.mp4 --output cam1_b.json --start 11600
python analyze.py --cam cam2.mp4 --output cam2_a.json --start 0     --end 11600
python analyze.py --cam cam2.mp4 --output cam2_b.json --start 11600

python merge_segments.py --inputs cam1_a.json cam1_b.json --output cam1.json
python merge_segments.py --inputs cam2_a.json cam2_b.json --output cam2.json
python render.py --cam1 cam1.mp4 --cam2 cam2.mp4 --json1 cam1.json --json2 cam2.json
```

### Speed by device

| Device | Model | skip | Time per camera |
|---|---|---|---|
| RTX 4090 | yolov8m | 1 | ~8 min |
| RTX 5090 Mobile | yolov8m | 1 | ~10 min |
| M4 / M5 (MPS) | yolov8m | 1 | ~20 min |
| CPU x86 | yolov8n | 2 | ~15 min |

With 2 machines running in parallel: total ≈ `max(cam1_time, cam2_time)`.  
With 4 machines: **~4–5 minutes** total.

---

## Setup

### Single machine (conda)

```bash
conda create -n basketball python=3.11 -y
conda activate basketball
pip install ultralytics deep_sort_realtime opencv-python tqdm PyQt6 "numpy<2" "setuptools<71"
```

### NVIDIA GPU (Windows / Linux)

Install PyTorch with CUDA **before** the rest:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install ultralytics deep_sort_realtime opencv-python tqdm PyQt6 "numpy<2" "setuptools<71"
```

### Apple Silicon M4 / M5 — native MPS

Use an `osx-arm64` conda env (not Rosetta) for MPS acceleration:

```bash
CONDA_SUBDIR=osx-arm64 conda create -n basketball python=3.11 -y
conda activate basketball
pip install ultralytics deep_sort_realtime opencv-python tqdm PyQt6 "numpy<2" "setuptools<71"
```

> Model weights download automatically on first run (~6 MB for yolov8n, ~50 MB for yolov8m).

---

## Files

| File | Purpose |
|---|---|
| `gui.py` | Visual editor — auto-analyze + manual cuts + export |
| `gui.sh` | Launch the editor via conda |
| `analyze.py` | CLI: single-camera YOLO+DeepSORT analysis → JSON |
| `render.py` | CLI: two JSONs + videos → annotated side-by-side video |
| `merge_segments.py` | Merge partial JSONs from segment-split runs |
| `run.sh` | Single-machine headless pipeline |
| `run_distributed.sh` | Distributed workflow reference |

---

## Requirements

- Python 3.11
- [ultralytics](https://github.com/ultralytics/ultralytics) — YOLOv8
- [deep_sort_realtime](https://github.com/levan92/deep_sort_realtime) — multi-object tracking
- OpenCV, NumPy, tqdm, PyQt6

GPU optional but strongly recommended — CUDA gives ~17× speedup over CPU.
