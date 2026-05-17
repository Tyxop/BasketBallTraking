"""
Basketball multi-camera tracker.
Analyzes cam1.mp4 and cam2.mp4 frame-by-frame, selects the "active" camera
per frame based on:
    ball detected > player touching ball > moving players > sitting players
Produces a side-by-side annotated output video.
"""

import cv2
import numpy as np
import json
import sys
import time
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field

from tqdm import tqdm
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ─── COCO class IDs ───────────────────────────────────────────────────────────
CLASS_PERSON = 0
CLASS_BALL   = 32   # "sports ball" in COCO

# ─── Ball detection ───────────────────────────────────────────────────────────
BALL_YOLO_CONF   = 0.25    # lower threshold helps with partial views
# HSV range for basketball orange (fallback color-based detector)
BALL_HSV_LOW     = np.array([5,  120, 120])
BALL_HSV_HIGH    = np.array([25, 255, 255])
BALL_MIN_AREA    = 200     # px²  — ignore tiny orange blobs
BALL_MAX_AREA    = 25000   # px²  — ignore huge orange regions (jerseys etc.)
BALL_MIN_CIRCULARITY = 0.45  # 1.0 = perfect circle

# ─── Scoring weights ──────────────────────────────────────────────────────────
SCORE_BALL_DETECTED    = 200
SCORE_PLAYER_WITH_BALL = 300
SCORE_MOVING_PLAYER    = 30
SCORE_SITTING_PLAYER   = 5

# ─── Motion classification ────────────────────────────────────────────────────
MOTION_HISTORY   = 15     # frames to average
MOVING_VEL_PX    = 3.5   # px/frame  (after resize to OUTPUT_H)
SITTING_W_H_RATIO = 0.70  # bbox width/height > this → likely sitting

# ─── Ball-player proximity ────────────────────────────────────────────────────
BALL_PLAYER_IOU_THRESH  = 0.01
BALL_PLAYER_DIST_FRAC   = 0.12   # fraction of frame height

# ─── Output layout ────────────────────────────────────────────────────────────
OUTPUT_W  = 1920
OUTPUT_H  = 540
BORDER_PX = 6

# ─── Colors BGR ───────────────────────────────────────────────────────────────
COL_MOVING  = (0, 220, 50)
COL_SITTING = (0, 165, 255)
COL_BALL    = (0, 80, 255)
COL_ACTIVE  = (0, 220, 50)
COL_IDLE    = (80, 80, 80)
COL_WHITE   = (255, 255, 255)


# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class TrackState:
    track_id: int
    centers: deque = field(default_factory=lambda: deque(maxlen=MOTION_HISTORY))

    def update(self, cx: float, cy: float):
        self.centers.append((cx, cy))

    @property
    def velocity(self) -> float:
        if len(self.centers) < 2:
            return 0.0
        pts = list(self.centers)
        dx = pts[-1][0] - pts[0][0]
        dy = pts[-1][1] - pts[0][1]
        return (dx**2 + dy**2) ** 0.5 / max(len(pts) - 1, 1)

    @property
    def is_moving(self) -> bool:
        return self.velocity >= MOVING_VEL_PX


# ──────────────────────────────────────────────────────────────────────────────
def iou(a, b) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / (ua + 1e-6)


def center_dist_frac(ball_box, player_box, fh: int) -> float:
    bc = ((ball_box[0]+ball_box[2])/2, (ball_box[1]+ball_box[3])/2)
    pc = ((player_box[0]+player_box[2])/2, (player_box[1]+player_box[3])/2)
    return ((bc[0]-pc[0])**2 + (bc[1]-pc[1])**2)**0.5 / max(fh, 1)


def player_near_ball(pbox, bball, fh: int) -> bool:
    return (iou(pbox, bball) >= BALL_PLAYER_IOU_THRESH or
            center_dist_frac(bball, pbox, fh) <= BALL_PLAYER_DIST_FRAC)


def is_sitting_shape(box) -> bool:
    w = box[2]-box[0]; h = box[3]-box[1]
    return h > 0 and (w / h) > SITTING_W_H_RATIO


# ──────────────────────────────────────────────────────────────────────────────
def detect_ball_color(frame: np.ndarray) -> list:
    """HSV-based basketball detector — returns list of [x1,y1,x2,y2] boxes."""
    blurred = cv2.GaussianBlur(frame, (9, 9), 0)
    hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask    = cv2.inRange(hsv, BALL_HSV_LOW, BALL_HSV_HIGH)
    mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
    mask    = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((5,5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (BALL_MIN_AREA < area < BALL_MAX_AREA):
            continue
        perim = cv2.arcLength(cnt, True)
        circ  = 4 * np.pi * area / (perim**2 + 1e-6)
        if circ < BALL_MIN_CIRCULARITY:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append([float(x), float(y), float(x+w), float(y+h)])
    return boxes


# ──────────────────────────────────────────────────────────────────────────────
class CameraAnalyzer:
    def __init__(self, model: YOLO):
        self.model   = model
        self.tracker = DeepSort(max_age=30, n_init=3, nn_budget=100)
        self.states: dict[int, TrackState] = {}

    def analyze(self, frame: np.ndarray) -> dict:
        h, w = frame.shape[:2]

        results = self.model(frame, classes=[CLASS_PERSON, CLASS_BALL],
                             conf=BALL_YOLO_CONF, verbose=False)[0]

        person_dets = []
        yolo_balls  = []
        for box in results.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            if cls == CLASS_BALL:
                yolo_balls.append([x1, y1, x2, y2])
            elif cls == CLASS_PERSON and conf >= 0.35:
                person_dets.append(([x1, y1, x2-x1, y2-y1], conf, cls))

        # Color fallback if YOLO misses the ball
        color_balls = detect_ball_color(frame)

        # Prefer YOLO ball; fallback to color if nothing found
        ball_boxes = yolo_balls if yolo_balls else color_balls
        ball_box   = ball_boxes[0] if ball_boxes else None
        ball_src   = ("yolo" if yolo_balls else "color") if ball_box else None

        # Tracking
        tracks = self.tracker.update_tracks(person_dets, frame=frame)

        moving  = []
        sitting = []
        carriers = set()

        for track in tracks:
            if not track.is_confirmed():
                continue
            tid       = track.track_id
            x1,y1,x2,y2 = track.to_ltrb()
            cx, cy    = (x1+x2)/2, (y1+y2)/2

            if tid not in self.states:
                self.states[tid] = TrackState(tid)
            st = self.states[tid]
            st.update(cx, cy)

            pbox = [x1, y1, x2, y2]
            is_sit = is_sitting_shape(pbox) or not st.is_moving
            entry  = {'id': tid, 'box': pbox, 'vel': st.velocity}

            (sitting if is_sit else moving).append(entry)
            if ball_box and player_near_ball(pbox, ball_box, h):
                carriers.add(tid)

        score = 0
        if ball_box:
            score += SCORE_BALL_DETECTED
        if carriers:
            score += SCORE_PLAYER_WITH_BALL
        score += len(moving)  * SCORE_MOVING_PLAYER
        score += len(sitting) * SCORE_SITTING_PLAYER

        return {
            'score':    score,
            'ball_box': ball_box,
            'ball_src': ball_src,
            'moving':   moving,
            'sitting':  sitting,
            'carriers': carriers,
            'h': h, 'w': w,
        }


# ──────────────────────────────────────────────────────────────────────────────
def annotate(frame: np.ndarray, info: dict, active: bool, label: str) -> np.ndarray:
    out = frame.copy()

    for p in info['moving']:
        x1,y1,x2,y2 = (int(v) for v in p['box'])
        has_ball = p['id'] in info['carriers']
        col = COL_BALL if has_ball else COL_MOVING
        cv2.rectangle(out, (x1,y1), (x2,y2), col, 2 + has_ball)
        txt = f"M#{p['id']}" + (" +BALL" if has_ball else "")
        cv2.putText(out, txt, (x1, max(y1-6,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

    for p in info['sitting']:
        x1,y1,x2,y2 = (int(v) for v in p['box'])
        cv2.rectangle(out, (x1,y1), (x2,y2), COL_SITTING, 1)
        cv2.putText(out, f"S#{p['id']}", (x1, max(y1-6,0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL_SITTING, 1)

    if info['ball_box']:
        x1,y1,x2,y2 = (int(v) for v in info['ball_box'])
        src_tag = f"BALL [{info['ball_src']}]"
        cv2.rectangle(out, (x1,y1), (x2,y2), COL_BALL, 3)
        cv2.putText(out, src_tag, (x1, max(y1-6,0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_BALL, 2)

    # Border
    h, w = out.shape[:2]
    cv2.rectangle(out, (0,0), (w-1,h-1), COL_ACTIVE if active else COL_IDLE, BORDER_PX)

    # HUD
    lines = [
        f"{label}  {'◀ ACTIVE' if active else ''}",
        f"Score: {info['score']}   Ball: {'YES' if info['ball_box'] else 'no'}",
        f"Moving: {len(info['moving'])}   Sitting: {len(info['sitting'])}",
    ]
    for i, ln in enumerate(lines):
        y = 24 + i * 24
        cv2.putText(out, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0,0,0), 4)
        cv2.putText(out, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, COL_WHITE, 2)

    return out


# ──────────────────────────────────────────────────────────────────────────────
def process(video1: str, video2: str, output: str,
            model_name: str = "yolov8m.pt", skip: int = 1):
    """
    skip=1 → every frame (slowest, most accurate)
    skip=2 → every other frame (2× faster)
    skip=3 → every 3rd frame, etc.
    """
    print(f"[INFO] Model: {model_name}  |  Frame skip: {skip}", flush=True)
    model = YOLO(model_name)

    cap1  = cv2.VideoCapture(video1)
    cap2  = cv2.VideoCapture(video2)
    fps1  = cap1.get(cv2.CAP_PROP_FPS) or 30
    fps2  = cap2.get(cv2.CAP_PROP_FPS) or 30
    fps   = min(fps1, fps2) / skip   # output fps matches real-time
    total = min(int(cap1.get(cv2.CAP_PROP_FRAME_COUNT)),
                int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))) // skip

    half_w = OUTPUT_W // 2
    writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (OUTPUT_W, OUTPUT_H))

    a1 = CameraAnalyzer(model)
    a2 = CameraAnalyzer(model)

    stats = dict(cam1_active=0, cam2_active=0,
                 cam1_ball=0,   cam2_ball=0,
                 total_frames=0)

    print(f"[INFO] ~{total} frames to process.", flush=True)

    bar = tqdm(
        total=total,
        unit="fr",
        dynamic_ncols=True,
        file=sys.stderr,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )

    try:
        while True:
            for _ in range(skip - 1):
                cap1.grab(); cap2.grab()

            ok1, f1 = cap1.read()
            ok2, f2 = cap2.read()
            if not ok1 or not ok2:
                break

            i1 = a1.analyze(f1)
            i2 = a2.analyze(f2)

            active = 1 if i1['score'] >= i2['score'] else 2
            stats['cam1_active' if active == 1 else 'cam2_active'] += 1
            if i1['ball_box']: stats['cam1_ball'] += 1
            if i2['ball_box']: stats['cam2_ball'] += 1
            stats['total_frames'] += 1

            ann1 = annotate(f1, i1, active == 1, "CAM 1")
            ann2 = annotate(f2, i2, active == 2, "CAM 2")

            p1 = cv2.resize(ann1, (half_w, OUTPUT_H), interpolation=cv2.INTER_LINEAR)
            p2 = cv2.resize(ann2, (half_w, OUTPUT_H), interpolation=cv2.INTER_LINEAR)
            writer.write(np.hstack([p1, p2]))

            ball1 = ("BALL:" + i1['ball_src']) if i1['ball_box'] else "no ball"
            ball2 = ("BALL:" + i2['ball_src']) if i2['ball_box'] else "no ball"
            bar.set_postfix_str(
                f"CAM{active} | C1 {i1['score']:3d}({ball1}) "
                f"C2 {i2['score']:3d}({ball2}) "
                f"mov={len(i1['moving'])}/{len(i2['moving'])}",
                refresh=False,
            )
            bar.update(1)

    except KeyboardInterrupt:
        tqdm.write("\n[WARN] Interrupted early — partial output written.")

    bar.close()
    cap1.release(); cap2.release(); writer.release()

    T = max(stats['total_frames'], 1)
    print("\n" + "="*60, flush=True)
    print("SUMMARY", flush=True)
    print("="*60, flush=True)
    print(f"  Frames processed  : {T}", flush=True)
    print(f"  CAM 1 active      : {stats['cam1_active']:5d}  ({100*stats['cam1_active']/T:.1f}%)", flush=True)
    print(f"  CAM 2 active      : {stats['cam2_active']:5d}  ({100*stats['cam2_active']/T:.1f}%)", flush=True)
    print(f"  CAM 1 ball seen   : {stats['cam1_ball']:5d}  ({100*stats['cam1_ball']/T:.1f}%)", flush=True)
    print(f"  CAM 2 ball seen   : {stats['cam2_ball']:5d}  ({100*stats['cam2_ball']/T:.1f}%)", flush=True)
    print(f"  Output            : {output}", flush=True)
    print("="*60, flush=True)

    out_path = Path(output)
    out_path.with_suffix(".json").write_text(json.dumps({
        **stats,
        'cam1_active_pct': round(100*stats['cam1_active']/T, 1),
        'cam2_active_pct': round(100*stats['cam2_active']/T, 1),
        'cam1_ball_pct':   round(100*stats['cam1_ball']/T, 1),
        'cam2_ball_pct':   round(100*stats['cam2_ball']/T, 1),
    }, indent=2))
    print(f"  Stats JSON        : {out_path.with_suffix('.json')}")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam1",   default="videos/cam1.mp4")
    ap.add_argument("--cam2",   default="videos/cam2.mp4")
    ap.add_argument("--output", default="output.mp4")
    ap.add_argument("--model",  default="yolov8m.pt",
                    choices=["yolov8n.pt","yolov8s.pt","yolov8m.pt","yolov8l.pt","yolov8x.pt"])
    ap.add_argument("--skip",   type=int, default=2,
                    help="Process every Nth frame (1=all, 2=half, 3=third…)")
    args = ap.parse_args()
    process(args.cam1, args.cam2, args.output, args.model, args.skip)
