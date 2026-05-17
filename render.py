"""
Phase 2 — render side-by-side annotated video from two JSON analysis files.
No ML required — pure OpenCV. Run on any machine after analyze.py finishes.
"""

import cv2
import json
import sys
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

OUTPUT_W  = 1920
OUTPUT_H  = 540
BORDER_PX = 6

COL_MOVING  = (0, 220, 50)
COL_SITTING = (0, 165, 255)
COL_BALL    = (0, 80, 255)
COL_ACTIVE  = (0, 220, 50)
COL_IDLE    = (80, 80, 80)
COL_WHITE   = (255, 255, 255)


def _annotate(frame, fdata: dict, active: bool, label: str) -> np.ndarray:
    out = frame.copy()
    carriers = set(fdata.get('carriers', []))

    for p in fdata.get('moving', []):
        x1,y1,x2,y2 = (int(v) for v in p['box'])
        has_ball = p['id'] in carriers
        col = COL_BALL if has_ball else COL_MOVING
        cv2.rectangle(out, (x1,y1), (x2,y2), col, 2 + has_ball)
        txt = f"M#{p['id']}" + (" +BALL" if has_ball else "")
        cv2.putText(out, txt, (x1, max(y1-6,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

    for p in fdata.get('sitting', []):
        x1,y1,x2,y2 = (int(v) for v in p['box'])
        cv2.rectangle(out, (x1,y1), (x2,y2), COL_SITTING, 1)
        cv2.putText(out, f"S#{p['id']}", (x1, max(y1-6,0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL_SITTING, 1)

    ball = fdata.get('ball')
    if ball:
        x1,y1,x2,y2 = (int(v) for v in ball)
        tag = f"BALL [{fdata.get('ball_src','')}]"
        cv2.rectangle(out, (x1,y1), (x2,y2), COL_BALL, 3)
        cv2.putText(out, tag, (x1, max(y1-6,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_BALL, 2)

    h, w = out.shape[:2]
    cv2.rectangle(out, (0,0), (w-1,h-1), COL_ACTIVE if active else COL_IDLE, BORDER_PX)

    lines = [
        f"{label}  {'◀ ACTIVE' if active else ''}",
        f"Score: {fdata.get('score',0)}   Ball: {'YES' if ball else 'no'}",
        f"Moving: {len(fdata.get('moving',[]))}   Sitting: {len(fdata.get('sitting',[]))}",
    ]
    for i, ln in enumerate(lines):
        y = 24 + i*24
        cv2.putText(out, ln, (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0,0,0), 4)
        cv2.putText(out, ln, (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, COL_WHITE, 2)

    return out


def render(cam1: str, cam2: str, json1: str, json2: str, output: str):
    print(f"[INFO] Loading analysis data…", flush=True)
    data1 = json.loads(Path(json1).read_text())
    data2 = json.loads(Path(json2).read_text())

    frames1 = {f['frame']: f for f in data1['frames']}
    frames2 = {f['frame']: f for f in data2['frames']}

    # Build a sorted list of frame indices present in both
    common = sorted(set(frames1) & set(frames2))
    if not common:
        print("[ERROR] No overlapping frames between the two JSONs.", flush=True)
        return

    fps  = data1['fps'] / data1['skip']
    half = OUTPUT_W // 2

    cap1 = cv2.VideoCapture(cam1)
    cap2 = cv2.VideoCapture(cam2)
    writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (OUTPUT_W, OUTPUT_H))

    stats = dict(cam1_active=0, cam2_active=0, cam1_ball=0, cam2_ball=0)

    print(f"[INFO] Rendering {len(common)} frames → {output}", flush=True)
    bar = tqdm(total=len(common), unit="fr", dynamic_ncols=True, file=sys.stdout,
               bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

    prev_idx = -1
    for frame_idx in common:
        # Seek only if not sequential
        skip = data1['skip']
        expected = prev_idx + skip
        if frame_idx != expected:
            cap1.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            cap2.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        prev_idx = frame_idx

        ok1, f1 = cap1.read()
        ok2, f2 = cap2.read()
        if not ok1 or not ok2:
            break

        fd1 = frames1[frame_idx]
        fd2 = frames2[frame_idx]
        active = 1 if fd1['score'] >= fd2['score'] else 2

        stats['cam1_active' if active==1 else 'cam2_active'] += 1
        if fd1['ball']: stats['cam1_ball'] += 1
        if fd2['ball']: stats['cam2_ball'] += 1

        ann1 = _annotate(f1, fd1, active==1, "CAM 1")
        ann2 = _annotate(f2, fd2, active==2, "CAM 2")

        p1 = cv2.resize(ann1, (half, OUTPUT_H), interpolation=cv2.INTER_LINEAR)
        p2 = cv2.resize(ann2, (half, OUTPUT_H), interpolation=cv2.INTER_LINEAR)
        writer.write(np.hstack([p1, p2]))
        bar.update(1)

    bar.close()
    cap1.release(); cap2.release(); writer.release()

    T = max(sum(stats.values()) // 2, 1)
    print(f"\n{'='*55}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*55}", flush=True)
    print(f"  CAM 1 active  : {stats['cam1_active']:5d}  ({100*stats['cam1_active']/T:.1f}%)", flush=True)
    print(f"  CAM 2 active  : {stats['cam2_active']:5d}  ({100*stats['cam2_active']/T:.1f}%)", flush=True)
    print(f"  CAM 1 ball    : {stats['cam1_ball']:5d}  ({100*stats['cam1_ball']/T:.1f}%)", flush=True)
    print(f"  CAM 2 ball    : {stats['cam2_ball']:5d}  ({100*stats['cam2_ball']/T:.1f}%)", flush=True)
    print(f"  Output        : {output}", flush=True)
    print(f"{'='*55}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Render side-by-side video from two analysis JSONs")
    ap.add_argument("--cam1",  required=True)
    ap.add_argument("--cam2",  required=True)
    ap.add_argument("--json1", required=True)
    ap.add_argument("--json2", required=True)
    ap.add_argument("--output", default="output.mp4")
    args = ap.parse_args()
    render(args.cam1, args.cam2, args.json1, args.json2, args.output)
