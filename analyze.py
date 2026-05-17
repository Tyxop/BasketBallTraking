"""
Phase 1 — single-camera analysis.
Runs YOLO + DeepSORT on one video and writes a compact JSON with
per-frame detections. Run this on each camera independently (in parallel
on different machines) and then feed both JSONs to render.py.

Supports CUDA (NVIDIA), MPS (Apple Silicon) and CPU automatically.
"""

import cv2
import json
import sys
import time
import argparse
import numpy as np
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field

from tqdm import tqdm
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ─── COCO ─────────────────────────────────────────────────────────────────────
CLASS_PERSON = 0
CLASS_BALL   = 32

# ─── Ball color detector (HSV orange) ─────────────────────────────────────────
BALL_HSV_LOW         = np.array([8,  150, 150])
BALL_HSV_HIGH        = np.array([22, 255, 255])
BALL_MIN_AREA        = 400
BALL_MAX_AREA        = 30000
BALL_MIN_CIRCULARITY = 0.58

# ─── Motion ───────────────────────────────────────────────────────────────────
MOTION_HISTORY    = 15
MOVING_VEL_PX     = 3.5
SITTING_W_H_RATIO = 0.70

# ─── Ball-player proximity ────────────────────────────────────────────────────
BALL_IOU_THRESH  = 0.01
BALL_DIST_FRAC   = 0.12


def _detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"[INFO] CUDA device: {name}", flush=True)
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("[INFO] MPS device (Apple Silicon)", flush=True)
            return "mps"
    except Exception:
        pass
    print("[INFO] No GPU found — using CPU", flush=True)
    return "cpu"


@dataclass
class TrackState:
    track_id: int
    centers: deque = field(default_factory=lambda: deque(maxlen=MOTION_HISTORY))

    def update(self, cx, cy):
        self.centers.append((cx, cy))

    @property
    def velocity(self) -> float:
        if len(self.centers) < 2:
            return 0.0
        pts = list(self.centers)
        dx, dy = pts[-1][0]-pts[0][0], pts[-1][1]-pts[0][1]
        return (dx**2 + dy**2)**0.5 / max(len(pts)-1, 1)

    @property
    def is_moving(self) -> bool:
        return self.velocity >= MOVING_VEL_PX


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0],b[0]), max(a[1],b[1])
    ix2, iy2 = min(a[2],b[2]), min(a[3],b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if not inter: return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / (ua + 1e-6)


def _dist_frac(ball, player, fh) -> float:
    bc = ((ball[0]+ball[2])/2, (ball[1]+ball[3])/2)
    pc = ((player[0]+player[2])/2, (player[1]+player[3])/2)
    return ((bc[0]-pc[0])**2 + (bc[1]-pc[1])**2)**0.5 / max(fh, 1)


def _near_ball(pbox, bball, fh) -> bool:
    return (_iou(pbox, bball) >= BALL_IOU_THRESH or
            _dist_frac(bball, pbox, fh) <= BALL_DIST_FRAC)


def _detect_ball_color(frame) -> list:
    fh   = frame.shape[0]
    blur = cv2.GaussianBlur(frame, (9,9), 0)
    hsv  = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BALL_HSV_LOW, BALL_HSV_HIGH)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((5,5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (BALL_MIN_AREA < area < BALL_MAX_AREA): continue
        perim = cv2.arcLength(cnt, True)
        if 4*np.pi*area/(perim**2+1e-6) < BALL_MIN_CIRCULARITY: continue
        x,y,w,h = cv2.boundingRect(cnt)
        if max(w,h) > 1.6 * min(w,h): continue          # reject elongated shapes
        if (y + h/2) < 0.12 * fh:     continue          # reject scoreboard/ceiling area
        boxes.append([float(x), float(y), float(x+w), float(y+h)])
    return boxes


def _is_sitting(box) -> bool:
    w, h = box[2]-box[0], box[3]-box[1]
    return h > 0 and (w/h) > SITTING_W_H_RATIO


def analyze(video: str, output_json: str, model_name: str, skip: int, segment: tuple):
    device = _detect_device()
    print(f"[INFO] Model: {model_name}  skip: {skip}  device: {device}", flush=True)
    model   = YOLO(model_name)
    model.to(device)
    tracker = DeepSort(max_age=30, n_init=3, nn_budget=100)
    states: dict[int, TrackState] = {}

    cap   = cv2.VideoCapture(video)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    total_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    seg_start, seg_end = segment
    if seg_end == -1:
        seg_end = total_raw
    seg_end = min(seg_end, total_raw)

    # Seek to segment start
    if seg_start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, seg_start)

    total = (seg_end - seg_start) // skip
    frames_data = []

    print(f"[INFO] Segment: frames {seg_start}–{seg_end}  ({(seg_end-seg_start)/fps:.1f}s)", flush=True)

    bar = tqdm(total=total, unit="fr", dynamic_ncols=True, file=sys.stdout,
               bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}")

    frame_pos = seg_start
    try:
        while frame_pos < seg_end:
            for _ in range(skip - 1):
                if frame_pos + 1 < seg_end:
                    cap.grab()
                    frame_pos += 1

            ok, frame = cap.read()
            frame_pos += 1
            if not ok or frame_pos > seg_end:
                break

            h, w = frame.shape[:2]
            results = model(frame, classes=[CLASS_PERSON, CLASS_BALL],
                            conf=0.25, verbose=False)[0]

            person_dets, yolo_balls = [], []
            for box in results.boxes:
                cls  = int(box.cls[0])
                conf = float(box.conf[0])
                x1,y1,x2,y2 = box.xyxy[0].tolist()
                if cls == CLASS_BALL:
                    yolo_balls.append([x1,y1,x2,y2])
                elif cls == CLASS_PERSON and conf >= 0.35:
                    person_dets.append(([x1, y1, x2-x1, y2-y1], conf, cls))

            color_balls = _detect_ball_color(frame)
            ball_boxes  = yolo_balls or color_balls
            ball_box    = ball_boxes[0] if ball_boxes else None
            ball_src    = ("yolo" if yolo_balls else "color") if ball_box else None

            tracks = tracker.update_tracks(person_dets, frame=frame)

            moving, sitting, carriers = [], [], set()
            for track in tracks:
                if not track.is_confirmed(): continue
                tid = track.track_id
                x1,y1,x2,y2 = track.to_ltrb()
                cx, cy = (x1+x2)/2, (y1+y2)/2
                if tid not in states:
                    states[tid] = TrackState(tid)
                states[tid].update(cx, cy)

                pbox = [x1,y1,x2,y2]
                is_sit = _is_sitting(pbox) or not states[tid].is_moving
                entry = {'id': tid, 'box': [round(v,1) for v in pbox],
                         'vel': round(states[tid].velocity, 2)}
                (sitting if is_sit else moving).append(entry)
                if ball_box and _near_ball(pbox, ball_box, h):
                    carriers.add(tid)

            # Scoring
            score = 0
            if ball_box:    score += 200
            if carriers:    score += 300
            score += len(moving)  * 30
            score += len(sitting) * 5

            frames_data.append({
                'frame':    frame_pos - 1,
                'score':    score,
                'ball':     [round(v,1) for v in ball_box] if ball_box else None,
                'ball_src': ball_src,
                'moving':   moving,
                'sitting':  sitting,
                'carriers': list(carriers),
            })

            ball_tag = f"BALL:{ball_src}" if ball_box else "no ball"
            bar.set_postfix_str(
                f"score={score} {ball_tag} mov={len(moving)} sit={len(sitting)}",
                refresh=False)
            bar.update(1)

    except KeyboardInterrupt:
        tqdm.write("\n[WARN] Interrupted — partial data saved.", file=sys.stdout)

    bar.close()
    cap.release()

    out = {
        'video':      str(video),
        'model':      model_name,
        'device':     device,
        'fps':        fps,
        'skip':       skip,
        'seg_start':  seg_start,
        'seg_end':    seg_end,
        'frames':     frames_data,
    }
    Path(output_json).write_text(json.dumps(out, separators=(',', ':')))
    size_mb = Path(output_json).stat().st_size / 1e6
    print(f"\n[OK] Saved {len(frames_data)} frames → {output_json}  ({size_mb:.1f} MB)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Single-camera basketball analysis → JSON")
    ap.add_argument("--cam",    required=True,         help="Input video path")
    ap.add_argument("--output", required=True,         help="Output JSON path")
    ap.add_argument("--model",  default="yolov8n.pt",
                    choices=["yolov8n.pt","yolov8s.pt","yolov8m.pt","yolov8l.pt","yolov8x.pt"])
    ap.add_argument("--skip",   type=int, default=2,   help="Process every Nth frame")
    ap.add_argument("--start",  type=int, default=0,   help="Start frame (for segment split)")
    ap.add_argument("--end",    type=int, default=-1,  help="End frame (-1 = until EOF)")
    args = ap.parse_args()

    analyze(args.cam, args.output, args.model, args.skip, (args.start, args.end))
