#!/usr/bin/env python3
"""
Basketball Editor — GUI para corte manual entre dos cámaras.

Uso:
    python gui.py
    python gui.py videos/cam1.mp4 videos/cam2.mp4
"""

import sys
import bisect
import os
import time
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QPoint
from PyQt6.QtGui import (QImage, QPixmap, QPainter, QColor, QPen,
                          QBrush, QFont, QPolygon, QKeySequence, QShortcut)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QFileDialog, QProgressDialog,
    QFrame, QSizePolicy, QMessageBox, QComboBox
)

# ── palette ───────────────────────────────────────────────────────────────────
DARK   = "#141414"
PANEL  = "#1e1e1e"
C1_HEX = "#4A9EDB"   # CAM 1 blue
C2_HEX = "#E8A040"   # CAM 2 orange
C1_COL = QColor(74,  158, 219)
C2_COL = QColor(232, 160,  64)

def _btn(bg, border, hover=""):
    hover = hover or bg
    return (f"QPushButton{{background:{bg};color:#eee;border:1px solid {border};"
            f"border-radius:5px;padding:7px 18px;font-weight:bold;}}"
            f"QPushButton:hover{{background:{hover};}}"
            f"QPushButton:disabled{{background:#2a2a2a;color:#555;}}")


# ── helpers ───────────────────────────────────────────────────────────────────

def _merge_ranges(ranges: list) -> list:
    """Une rangos solapados y los ordena."""
    if not ranges:
        return []
    out = [list(sorted(ranges)[0])]
    for s, e in sorted(ranges)[1:]:
        if s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [tuple(r) for r in out]


def fmt(s: float) -> str:
    m, sec = divmod(int(max(s, 0)), 60)
    return f"{m:02d}:{sec:02d}"


def to_pixmap(frame, size) -> QPixmap:
    if frame is None:
        return QPixmap()
    rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, c = rgb.shape
    img  = QImage(rgb.data, w, h, w * c, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(img).scaled(
        size, Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation)


# ── data model ────────────────────────────────────────────────────────────────

class CutList:
    """
    Sorted list of (time_sec, cam_id).
    Always starts with (0.0, cam_id) meaning «use this cam from the start».
    """

    def __init__(self):
        self._cuts   = [(0.0, 1)]
        self._history: list[list] = []

    @property
    def cuts(self) -> list:
        return list(self._cuts)

    def active_at(self, t: float) -> int:
        cam = self._cuts[0][1]
        for ct, c in self._cuts:
            if ct <= t:
                cam = c
            else:
                break
        return cam

    def add(self, t: float, cam: int):
        self._history.append(list(self._cuts))
        if t < 0.3:
            # Change the initial camera
            self._cuts[0] = (0.0, cam)
        else:
            # Remove any cut within ±0.3 s (but keep t=0)
            self._cuts = [(ct, c) for ct, c in self._cuts
                          if ct == 0.0 or abs(ct - t) > 0.3]
            bisect.insort(self._cuts, (t, cam))
        self._collapse()

    def undo(self):
        if self._history:
            self._cuts = self._history.pop()

    def _collapse(self):
        """Remove consecutive cuts with the same camera."""
        out = [self._cuts[0]]
        for ct, c in self._cuts[1:]:
            if c != out[-1][1]:
                out.append((ct, c))
        self._cuts = out

    def n_cuts(self) -> int:
        """User-visible number of transitions (not counting the initial one)."""
        return len(self._cuts) - 1

    def set_range(self, start: float, end: float, cam: int):
        """Asigna una cámara a todo el rango [start, end]."""
        self._history.append(list(self._cuts))
        cam_after = self.active_at(end)          # guardar antes de modificar
        # Eliminar cortes dentro del rango (excepto t=0)
        self._cuts = [(t, c) for t, c in self._cuts
                      if t == 0.0 or not (start <= t <= end)]
        # Poner la cámara al inicio del rango
        if start < 0.3:
            self._cuts[0] = (0.0, cam)
        else:
            bisect.insort(self._cuts, (start, cam))
        # Restaurar la cámara original al final del rango
        if cam_after != cam:
            bisect.insort(self._cuts, (end, cam_after))
        self._collapse()

    def clear_range(self, start: float, end: float):
        """Elimina todos los puntos de corte dentro de (start, end)."""
        self._history.append(list(self._cuts))
        self._cuts = [(t, c) for t, c in self._cuts
                      if t == 0.0 or not (start < t < end)]
        self._collapse()


# ── timeline widget ───────────────────────────────────────────────────────────

class TimelineWidget(QWidget):
    seek_requested    = pyqtSignal(float)
    section_selected  = pyqtSignal(float, float)   # (start, end)

    def __init__(self):
        super().__init__()
        self.duration     = 1.0
        self.position     = 0.0
        self.cuts: list[tuple[float, int]] = [(0.0, 1)]
        self._deleted: list[tuple[float, float]] = []
        self._select_mode = False
        self._sel_start: float | None = None
        self._sel_end:   float | None = None
        self.setMinimumHeight(72)
        self.setMaximumHeight(88)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_state(self, duration, position, cuts):
        self.duration = max(duration, 1.0)
        self.position = position
        self.cuts     = cuts
        self.update()

    def set_deleted(self, ranges):
        self._deleted = list(ranges)
        self.update()

    def set_select_mode(self, on: bool):
        self._select_mode = on
        self._sel_start = self._sel_end = None
        self.setCursor(Qt.CursorShape.CrossCursor if on
                       else Qt.CursorShape.PointingHandCursor)
        self.update()

    def clear_selection(self):
        self._sel_start = self._sel_end = None
        self.update()

    def _x(self, t: float) -> int:
        return int(t / self.duration * (self.width() - 1))

    def _t_at(self, ev) -> float:
        t = ev.position().x() / self.width() * self.duration
        return max(0.0, min(t, self.duration))

    def paintEvent(self, _ev):
        p  = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        BY, BH = 16, 32

        # Background
        p.fillRect(0, 0, W, H, QColor(20, 20, 20))

        # Colored segments
        for i, (ct, cam) in enumerate(self.cuts):
            nt  = self.cuts[i + 1][0] if i + 1 < len(self.cuts) else self.duration
            x1  = self._x(ct); x2 = self._x(nt)
            col = C1_COL if cam == 1 else C2_COL
            p.fillRect(x1, BY, max(x2 - x1, 1), BH, col)

        # Deleted ranges — overlay rojo oscuro + rayas diagonales
        for ds, de in self._deleted:
            x1, x2 = self._x(ds), self._x(de)
            w = max(x2 - x1, 1)
            p.fillRect(x1, BY, w, BH, QColor(120, 20, 20, 200))
            p.setPen(QPen(QColor(200, 50, 50, 120), 1))
            step = 6
            for xi in range(0, w + BH, step):
                p.drawLine(x1 + xi, BY, x1 + xi - BH, BY + BH)
            p.setPen(Qt.PenStyle.NoPen)

        # Cut-point markers
        p.setPen(QPen(QColor(255, 255, 255, 210), 1))
        for ct, _ in self.cuts[1:]:
            x = self._x(ct)
            p.drawLine(x, BY - 5, x, BY + BH + 5)

        # Time labels
        p.setFont(QFont("monospace", 8))
        p.setPen(QPen(QColor(140, 140, 140)))
        step = max(30, round(self.duration / 8 / 30) * 30)
        t = 0.0
        while t <= self.duration + 0.1:
            x = self._x(t)
            p.drawText(x + 2, BY + BH + 16, fmt(t))
            t += step

        # Selección activa — overlay amarillo semitransparente
        if self._sel_start is not None and self._sel_end is not None:
            a, b = sorted([self._sel_start, self._sel_end])
            if b > a:
                x1, x2 = self._x(a), self._x(b)
                p.fillRect(x1, BY - 3, max(x2 - x1, 1), BH + 6,
                           QColor(255, 200, 0, 80))
                p.setPen(QPen(QColor(255, 200, 0), 2))
                p.drawRect(x1, BY - 3, max(x2 - x1, 1), BH + 6)

        # Playhead
        px = self._x(self.position)
        p.setPen(QPen(QColor(255, 255, 255), 2))
        p.drawLine(px, BY - 8, px, BY + BH + 8)
        p.setBrush(QBrush(QColor(255, 255, 255)))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawPolygon(QPolygon([QPoint(px, BY+6),
                                QPoint(px-6, BY-2),
                                QPoint(px+6, BY-2)]))

        # Legend
        p.setFont(QFont("sans-serif", 8))
        for i, (lbl, col) in enumerate([("CAM 1", C1_COL), ("CAM 2", C2_COL)]):
            lx = W - 130 + i * 65
            p.fillRect(lx, 3, 12, 10, col)
            p.setPen(QPen(QColor(200, 200, 200)))
            p.drawText(lx + 14, 12, lbl)

        p.end()

    def mousePressEvent(self, ev):
        if self._select_mode and ev.button() == Qt.MouseButton.LeftButton:
            self._sel_start = self._sel_end = self._t_at(ev)
            self.update()
        else:
            self._emit_seek(ev)

    def mouseMoveEvent(self, ev):
        if self._select_mode and ev.buttons() & Qt.MouseButton.LeftButton:
            self._sel_end = self._t_at(ev)
            self.update()
        elif not self._select_mode and ev.buttons() & Qt.MouseButton.LeftButton:
            self._emit_seek(ev)

    def mouseReleaseEvent(self, ev):
        if self._select_mode and self._sel_start is not None:
            a, b = sorted([self._sel_start, self._sel_end or self._sel_start])
            if b - a > 0.1:
                self.section_selected.emit(a, b)

    def _emit_seek(self, ev):
        t = ev.position().x() / self.width() * self.duration
        self.seek_requested.emit(max(0.0, min(t, self.duration)))


# ── audio sync thread ────────────────────────────────────────────────────────

class SyncThread(QThread):
    """
    Extracts mono audio from both videos via ffmpeg, computes the
    FFT cross-correlation and emits the time offset (seconds) of
    cam2 relative to cam1.

    Positive offset  → cam2 started EARLIER than cam1
                       (to align: read cam2 at  t + offset)
    Negative offset  → cam2 started LATER than cam1
                       (to align: read cam2 at  t + offset, which is < t)
    """
    done   = pyqtSignal(float)   # offset_sec
    failed = pyqtSignal(str)
    status = pyqtSignal(str)

    SR = 8000   # sample rate — low enough to be fast, high enough for accuracy

    def __init__(self, cam1: str, cam2: str):
        super().__init__()
        self.cam1, self.cam2 = cam1, cam2

    def run(self):
        try:
            self.status.emit("Extrayendo audio de CAM 1…")
            a1 = self._extract(self.cam1)
            self.status.emit("Extrayendo audio de CAM 2…")
            a2 = self._extract(self.cam2)
            self.status.emit("Calculando correlación…")

            # Normalize to unit variance
            a1 = a1 / (np.std(a1) + 1e-9)
            a2 = a2 / (np.std(a2) + 1e-9)

            from scipy.signal import correlate
            corr = correlate(a1, a2, mode="full", method="fft")
            lag  = int(np.argmax(np.abs(corr))) - (len(a2) - 1)
            offset_sec = lag / self.SR
            self.done.emit(float(offset_sec))

        except Exception:
            import traceback
            self.failed.emit(traceback.format_exc())

    def _extract(self, path: str) -> np.ndarray:
        cmd = [
            "ffmpeg", "-y", "-i", path,
            "-ac", "1",                   # mono
            "-ar", str(self.SR),          # resample
            "-f", "f32le",                # raw float32
            "-vn",                        # no video
            "pipe:1",
        ]
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.decode(errors="replace")[-800:])
        return np.frombuffer(r.stdout, dtype=np.float32).copy()


# ── fast IoU tracker (replaces DeepSORT, zero GPU overhead) ──────────────────

class _FastTracker:
    """Greedy IoU tracker (SORT-style). No CNN, ~1 µs per frame."""

    def __init__(self, max_age=30, n_init=3, iou_thresh=0.3):
        self.max_age    = max_age
        self.n_init     = n_init
        self.iou_thresh = iou_thresh
        self._tracks    = {}   # tid → {box, hits, age}
        self._next_id   = 1

    @staticmethod
    def _iou(a, b):
        ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if not inter:
            return 0.0
        ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / (ua + 1e-6)

    def update(self, boxes):
        """boxes: [[x1,y1,x2,y2], …]. Returns confirmed [(tid, box), …]."""
        for tid in list(self._tracks):
            self._tracks[tid]['age'] += 1
            if self._tracks[tid]['age'] > self.max_age:
                del self._tracks[tid]

        track_ids = list(self._tracks)

        if boxes and track_ids:
            tb = [self._tracks[tid]['box'] for tid in track_ids]
            pairs = sorted(
                [(self._iou(tb[ti], boxes[di]), ti, di)
                 for ti in range(len(track_ids))
                 for di in range(len(boxes))],
                reverse=True)
            matched_t, matched_d = set(), set()
            for iou_val, ti, di in pairs:
                if iou_val < self.iou_thresh:
                    break
                if ti in matched_t or di in matched_d:
                    continue
                matched_t.add(ti); matched_d.add(di)
                tid = track_ids[ti]
                t   = self._tracks[tid]
                t['box'] = boxes[di]; t['hits'] += 1; t['age'] = 0
            for di in range(len(boxes)):
                if di not in matched_d:
                    self._tracks[self._next_id] = {'box': boxes[di], 'hits': 1, 'age': 0}
                    self._next_id += 1
        else:
            for box in boxes:
                self._tracks[self._next_id] = {'box': box, 'hits': 1, 'age': 0}
                self._next_id += 1

        return [(tid, t['box']) for tid, t in self._tracks.items()
                if t['hits'] >= self.n_init]


# ── auto-analyze thread ───────────────────────────────────────────────────────

class AutoAnalyzeThread(QThread):
    """
    Runs YOLO + DeepSORT on both cameras and converts per-frame scores
    into a list of (time_sec, cam_id) cuts with smoothing + hysteresis.
    """
    progress = pyqtSignal(int, int, str)   # current_frame, total, status
    done     = pyqtSignal(list)            # [(time_sec, cam_id), ...]
    failed   = pyqtSignal(str)

    def __init__(self, cam1, cam2, fps, total_frames,
                 model_name="yolov8n.pt", skip=2,
                 smooth_sec=2.0, min_dur_sec=3.0):
        super().__init__()
        self.cam1, self.cam2      = cam1, cam2
        self.fps, self.total      = fps, total_frames
        self.model_name, self.skip = model_name, skip
        self.smooth_sec           = smooth_sec
        self.min_dur_sec          = min_dur_sec
        self._abort               = False

    def abort(self):
        self._abort = True

    # ── main work ─────────────────────────────────────────────────────────────

    # ── per-frame analysis helpers (inlined to avoid cross-file imports) ────────

    @staticmethod
    def _ball_color(frame):
        low  = np.array([5,  120, 120])
        high = np.array([25, 255, 255])
        blur = cv2.GaussianBlur(frame, (9, 9), 0)
        mask = cv2.inRange(cv2.cvtColor(blur, cv2.COLOR_BGR2HSV), low, high)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((5, 5), np.uint8))
        boxes = []
        for cnt in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            area = cv2.contourArea(cnt)
            if not (200 < area < 25000):
                continue
            perim = cv2.arcLength(cnt, True)
            if 4 * np.pi * area / (perim ** 2 + 1e-6) < 0.45:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append([float(x), float(y), float(x + w), float(y + h)])
        return boxes

    @staticmethod
    def _score_from_results(yolo_result, frame, tracker, states, precomp_ball=None):
        from collections import deque
        h = frame.shape[0]

        person_boxes, yolo_balls = [], []
        for box in yolo_result.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            if cls == 32:
                yolo_balls.append([x1, y1, x2, y2])
            elif cls == 0 and conf >= 0.35:
                person_boxes.append([x1, y1, x2, y2])

        ball_list = yolo_balls or precomp_ball or []
        ball = ball_list[0] if ball_list else None

        confirmed = tracker.update(person_boxes)
        moving    = 0
        sitting   = 0
        has_ball  = False

        for tid, (x1, y1, x2, y2) in confirmed:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if tid not in states:
                states[tid] = deque(maxlen=15)
            states[tid].append((cx, cy))

            pts = list(states[tid])
            vel = 0.0
            if len(pts) >= 2:
                dx, dy = pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1]
                vel = (dx**2 + dy**2)**0.5 / max(len(pts) - 1, 1)

            w = x2 - x1; ht = y2 - y1
            is_sit = (ht > 0 and w / ht > 0.70) or vel < 3.5
            if is_sit:
                sitting += 1
            else:
                moving += 1

            if ball:
                bx, by = (ball[0]+ball[2])/2, (ball[1]+ball[3])/2
                px, py = cx, cy
                if ((bx-px)**2 + (by-py)**2)**0.5 / max(h, 1) < 0.12:
                    has_ball = True

        score = 0
        if ball:      score += 200
        if has_ball:  score += 300
        score += moving  * 30
        score += sitting * 5
        return score

    def run(self):
        try:
            self.progress.emit(0, 1, "Cargando modelo…")
            import queue
            import threading
            import torch
            from ultralytics import YOLO

            device = "cpu"
            try:
                if torch.cuda.is_available():
                    device = "cuda"
                    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}", flush=True)
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = "mps"
            except Exception:
                pass

            model = YOLO(self.model_name)
            model.to(device)

            half       = (device == "cuda")
            BATCH_MIN  = 4
            BATCH_MAX  = 256
            BATCH_P    = BATCH_MIN
            QSZ        = BATCH_MAX * 4
            GPU_TARGET = 0.80
            dev_label  = device.upper()

            track1 = _FastTracker(max_age=30, n_init=3)
            track2 = _FastTracker(max_age=30, n_init=3)
            states1: dict = {}
            states2: dict = {}

            # ── detección de NVDEC ────────────────────────────────────────────
            use_nvdec = False
            if device == "cuda":
                try:
                    r = subprocess.run(['ffmpeg', '-hwaccels'],
                                       capture_output=True, text=True, timeout=5)
                    use_nvdec = 'cuda' in r.stdout
                except Exception:
                    pass
            print(f"[INFO] Decode: {'NVDEC (ffmpeg)' if use_nvdec else 'cv2/MSMF'}", flush=True)

            # Dimensiones de los vídeos (necesario para leer bytes crudos de ffmpeg)
            def _wh(path):
                c = cv2.VideoCapture(path)
                w, h = int(c.get(cv2.CAP_PROP_FRAME_WIDTH)), int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))
                c.release(); return w, h

            W1, H1 = _wh(self.cam1)
            W2, H2 = _wh(self.cam2)

            total = self.total // self.skip
            raw   = []
            t0    = time.time()
            t_gpu = 0.0

            # ── hilos de decodificación ───────────────────────────────────────
            q1: queue.Queue = queue.Queue(maxsize=QSZ)
            q2: queue.Queue = queue.Queue(maxsize=QSZ)

            def _ffmpeg_nvdec(path, W, H, q):
                """Decodifica con NVDEC y envía frames como numpy BGR."""
                frame_bytes = W * H * 3
                max_frames  = self.total // self.skip
                sf = f"select=not(mod(n,{self.skip}))" if self.skip > 1 else "select=1"
                cmd = ['ffmpeg', '-loglevel', 'error',
                       '-hwaccel', 'cuda',
                       '-i', path,
                       '-vf', sf, '-vsync', '0',
                       '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-']
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                        stderr=subprocess.DEVNULL,
                                        bufsize=frame_bytes * 4)
                try:
                    for _ in range(max_frames):
                        if self._abort: break
                        data = proc.stdout.read(frame_bytes)
                        if len(data) < frame_bytes: break
                        q.put(np.frombuffer(data, np.uint8).reshape(H, W, 3).copy())
                finally:
                    try: proc.kill(); proc.wait()
                    except Exception: pass
                q.put(None)

            def _cv2_reader(path, q):
                """Fallback: decodificación software con cv2 + MSMF."""
                cap = cv2.VideoCapture(path, cv2.CAP_MSMF)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(path)
                idx = 0
                while idx * self.skip < self.total and not self._abort:
                    for _ in range(self.skip - 1): cap.grab()
                    ok, f = cap.read()
                    if not ok: break
                    q.put(f); idx += 1
                cap.release()
                q.put(None)

            if use_nvdec:
                threading.Thread(target=_ffmpeg_nvdec, args=(self.cam1, W1, H1, q1), daemon=True).start()
                threading.Thread(target=_ffmpeg_nvdec, args=(self.cam2, W2, H2, q2), daemon=True).start()
            else:
                threading.Thread(target=_cv2_reader, args=(self.cam1, q1), daemon=True).start()
                threading.Thread(target=_cv2_reader, args=(self.cam2, q2), daemon=True).start()

            import concurrent.futures
            N_WORKERS = max(1, (os.cpu_count() or 4) - 2)  # deja 2 cores para decoders
            ball_pool = concurrent.futures.ThreadPoolExecutor(max_workers=N_WORKERS)

            self.progress.emit(0, total, f"Analizando… [{dev_label}{'  FP16' if half else ''}]")

            buf_f1, buf_f2, idx = [], [], 0

            while True:
                if self._abort:
                    break

                f1 = q1.get()
                f2 = q2.get()
                eof = (f1 is None or f2 is None)

                if not eof:
                    buf_f1.append(f1)
                    buf_f2.append(f2)

                if (len(buf_f1) >= BATCH_P) or (eof and buf_f1):
                    flat = [f for ab in zip(buf_f1, buf_f2) for f in ab]

                    # _ball_color en paralelo (N_WORKERS cores) solapado con YOLO GPU
                    ball_futs = [ball_pool.submit(self._ball_color, f) for f in flat]

                    tg = time.perf_counter()
                    results = model(flat, classes=[0, 32], conf=0.25,
                                    verbose=False, half=half)
                    t_gpu += time.perf_counter() - tg

                    balls = [fut.result() for fut in ball_futs]

                    for i in range(len(buf_f1)):
                        s1 = self._score_from_results(results[i*2],   buf_f1[i], track1, states1, balls[i*2])
                        s2 = self._score_from_results(results[i*2+1], buf_f2[i], track2, states2, balls[i*2+1])
                        raw.append((s1, s2))
                    idx += len(buf_f1)
                    buf_f1.clear(); buf_f2.clear()

                    # ── ajuste dinámico de batch ──────────────────────────────
                    t_elapsed = time.time() - t0
                    if device == "cuda" and t_elapsed > 2.0:
                        gpu_frac = t_gpu / t_elapsed
                        if gpu_frac < GPU_TARGET and BATCH_P < BATCH_MAX:
                            BATCH_P = min(int(BATCH_P * 1.5) + 2, BATCH_MAX)
                        elif gpu_frac > 0.97 and BATCH_P > BATCH_MIN:
                            BATCH_P = max(BATCH_P - 1, BATCH_MIN)

                    fps_real = idx / max(t_elapsed, 0.01)
                    eta      = (total - idx) / max(fps_real, 0.01)
                    gpu_pct  = int(t_gpu / max(t_elapsed, 0.001) * 100)
                    self.progress.emit(
                        idx, total,
                        f"Analizando… {idx}/{total}  {fps_real:.1f} fr/s  "
                        f"batch={BATCH_P}  GPU={gpu_pct}%  "
                        f"ETA {eta/60:.1f} min  [{dev_label}]")

                if eof:
                    break

            ball_pool.shutdown(wait=False)

            if self._abort:
                return

            self.progress.emit(total, total, "Calculando cortes…")
            cuts = self._scores_to_cuts(raw)
            self.done.emit(cuts)

        except Exception:
            import traceback
            self.failed.emit(traceback.format_exc())

    # ── score → cuts algorithm ────────────────────────────────────────────────

    def _scores_to_cuts(self, raw):
        n = len(raw)
        if n == 0:
            return [(0.0, 1)]

        # Rolling-window smoothing
        win  = max(1, int(self.smooth_sec  * self.fps / self.skip))
        minf = max(1, int(self.min_dur_sec * self.fps / self.skip))
        sm1, sm2 = [], []
        s1 = s2 = 0.0
        for i, (a, b) in enumerate(raw):
            s1 += a; s2 += b
            if i >= win:
                s1 -= raw[i - win][0]
                s2 -= raw[i - win][1]
            cnt = min(i + 1, win)
            sm1.append(s1 / cnt)
            sm2.append(s2 / cnt)

        desired = [1 if sm1[i] >= sm2[i] else 2 for i in range(n)]

        # Hysteresis: only switch after new camera dominates >= minf frames
        cuts        = [(0.0, desired[0])]
        current_cam = desired[0]
        cand_cam    = None
        cand_count  = 0

        for i in range(1, n):
            d = desired[i]
            if d == current_cam:
                cand_cam = None; cand_count = 0
            else:
                if d == cand_cam:
                    cand_count += 1
                    if cand_count >= minf:
                        t = (i - cand_count + 1) * self.skip / self.fps
                        if cuts[-1][1] != d:
                            cuts.append((round(t, 2), d))
                        current_cam = d
                        cand_cam = None; cand_count = 0
                else:
                    cand_cam = d; cand_count = 1

        return cuts


# ── background generator ──────────────────────────────────────────────────────

class GeneratorThread(QThread):
    progress = pyqtSignal(int, int)
    done     = pyqtSignal(str)
    failed   = pyqtSignal(str)

    def __init__(self, cam1, cam2, output, cuts, fps, total,
                 cam2_offset_frames: int = 0, total_frames2: int = 0,
                 deleted_ranges=None):
        super().__init__()
        self.cam1, self.cam2 = cam1, cam2
        self.output, self.cuts, self.fps, self.total = output, cuts, fps, total
        self.cam2_offset_frames = cam2_offset_frames
        self.total_frames2      = total_frames2 or total
        self.deleted_ranges     = deleted_ranges or []
        self._abort = False

    def abort(self):
        self._abort = True

    def _active_at(self, t):
        cam = self.cuts[0][1]
        for ct, c in self.cuts:
            if ct <= t: cam = c
            else: break
        return cam

    @staticmethod
    def _detect_encoder():
        """Devuelve 'h264_nvenc', 'hevc_nvenc' o None si no hay NVENC."""
        try:
            r = subprocess.run(['ffmpeg', '-encoders'],
                               capture_output=True, text=True, timeout=5)
            if 'h264_nvenc' in r.stdout:
                return 'h264_nvenc'
        except Exception:
            pass
        return None

    def run(self):
        try:
            cap1 = cv2.VideoCapture(self.cam1)
            cap2 = cv2.VideoCapture(self.cam2)
            W  = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
            H  = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
            W2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
            H2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # ── encoder: NVENC si disponible, mp4v como fallback ─────────────
            nvenc = self._detect_encoder()
            if nvenc:
                enc_cmd = [
                    'ffmpeg', '-y', '-loglevel', 'error',
                    '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                    '-s', f'{W}x{H}', '-r', str(self.fps),
                    '-i', 'pipe:0',
                    '-c:v', nvenc,
                    '-preset', 'p4',   # balance velocidad/calidad
                    '-cq', '23',       # calidad constante (0–51, menor = mejor)
                    '-pix_fmt', 'yuv420p',
                    self.output
                ]
                enc_proc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE,
                                            stderr=subprocess.DEVNULL)
                def _write(frame):
                    enc_proc.stdin.write(frame.tobytes())
                def _close():
                    enc_proc.stdin.close()
                    enc_proc.wait()
                print(f"[INFO] Encoder: {nvenc}", flush=True)
            else:
                writer = cv2.VideoWriter(
                    self.output, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (W, H))
                def _write(frame):
                    writer.write(frame)
                def _close():
                    writer.release()
                print("[INFO] Encoder: mp4v (software)", flush=True)

            # Pre-seek cam2 to its aligned starting position
            off = self.cam2_offset_frames
            if off > 0:
                cap2.set(cv2.CAP_PROP_POS_FRAMES, off)
            elif off < 0:
                cap1.set(cv2.CAP_PROP_POS_FRAMES, -off)

            prev_cam2_n = off - 1

            for n in range(self.total):
                if self._abort:
                    break

                cam1_time = (n + max(0, -off)) / self.fps
                active    = self._active_at(cam1_time)

                ok1, f1 = cap1.read()

                cam2_n = n + off
                cam2_n = max(0, min(cam2_n, self.total_frames2 - 1))
                if cam2_n != prev_cam2_n + 1:
                    cap2.set(cv2.CAP_PROP_POS_FRAMES, cam2_n)
                ok2, f2   = cap2.read()
                prev_cam2_n = cam2_n

                # Saltar frames en rangos borrados
                if any(ds <= cam1_time < de for ds, de in self.deleted_ranges):
                    if n % 120 == 0:
                        self.progress.emit(n, self.total)
                    continue

                frame = f1 if active == 1 else f2
                if frame is not None:
                    if active == 2 and (W2 != W or H2 != H):
                        frame = cv2.resize(frame, (W, H))
                    _write(frame)
                if n % 120 == 0:
                    self.progress.emit(n, self.total)

            cap1.release(); cap2.release()
            _close()
            if not self._abort:
                self.done.emit(self.output)
        except Exception as e:
            self.failed.emit(str(e))


# ── main window ───────────────────────────────────────────────────────────────

class BasketballEditor(QMainWindow):

    def __init__(self, cam1_path=None, cam2_path=None):
        super().__init__()
        self.setWindowTitle("Basketball Editor")
        self.setMinimumSize(1280, 780)
        self.setStyleSheet(f"QMainWindow,QWidget{{background:{DARK};color:#ddd;}}")

        self.cap1 = self.cap2 = None
        self.cam1_path = self.cam2_path = ""
        self.fps = 30.0
        self.total_frames  = 0
        self.total_frames2 = 0
        self.duration = 0.0
        self.current_frame = 0
        self.playing = False
        self.cuts = CutList()
        self._gen: GeneratorThread | None = None
        # Audio sync
        self.cam2_offset_sec    = 0.0   # cam2 local time = cam1 time + offset
        self.cam2_offset_frames = 0     # = round(offset_sec * fps)
        self._last_cam2_frame   = -999  # tracks sequential reading of cap2
        self.deleted_ranges: list[tuple[float, float]] = []
        self._deleted_history: list = []
        self._current_sel: tuple[float, float] | None = None

        self._build_ui()
        self._bind_shortcuts()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)

        self._set_controls_enabled(False)
        if cam1_path and cam2_path:
            self._load(cam1_path, cam2_path)

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        vbox = QVBoxLayout(root)
        vbox.setSpacing(6)
        vbox.setContentsMargins(10, 10, 10, 10)

        # Video panels
        row_vid = QHBoxLayout()
        row_vid.setSpacing(8)
        self.pnl1 = _VideoPanel("CAM 1", C1_HEX)
        self.pnl2 = _VideoPanel("CAM 2", C2_HEX)
        row_vid.addWidget(self.pnl1, 1)
        row_vid.addWidget(self.pnl2, 1)
        vbox.addLayout(row_vid, stretch=5)

        # Sync bar
        row_sync = QHBoxLayout()
        row_sync.setSpacing(10)

        self.btn_sync = QPushButton("🔊  Sincronizar audio")
        self.btn_sync.setToolTip(
            "Extrae el audio de ambas cámaras y calcula el desfase\n"
            "buscando el mismo pico de sonido (correlación cruzada).")
        self.btn_sync.clicked.connect(self._sync_audio)
        self.btn_sync.setStyleSheet(_btn("#1a2a3a", "#4080b0", "#223344"))

        self.lbl_offset = QLabel("Desfase: sin sincronizar")
        self.lbl_offset.setStyleSheet(
            "color:#aaa; font-family:monospace; font-size:12px;")

        row_sync.addWidget(self.btn_sync)
        row_sync.addWidget(self.lbl_offset)
        row_sync.addStretch()
        vbox.addLayout(row_sync)

        # Timeline
        self.timeline = TimelineWidget()
        self.timeline.seek_requested.connect(self._seek_sec)
        self.timeline.section_selected.connect(self._on_section_selected)
        vbox.addWidget(self.timeline)

        # Playback controls
        row_ctrl = QHBoxLayout()
        self.btn_play = QPushButton("▶")
        self.btn_play.setFixedWidth(46)
        self.btn_play.setStyleSheet(_btn("#2a2a2a", "#555", "#383838"))
        self.btn_play.clicked.connect(self._toggle_play)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.sliderMoved.connect(self._slider_moved)
        self.slider.setStyleSheet(
            "QSlider::groove:horizontal{height:4px;background:#3a3a3a;border-radius:2px;}"
            "QSlider::handle:horizontal{width:14px;height:14px;margin:-5px 0;"
            "background:white;border-radius:7px;}"
            f"QSlider::sub-page:horizontal{{background:{C1_HEX};border-radius:2px;}}")

        self.lbl_time = QLabel("00:00 / 00:00")
        self.lbl_time.setStyleSheet("color:#888;font-family:monospace;min-width:110px;")
        self.lbl_time.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        row_ctrl.addWidget(self.btn_play)
        row_ctrl.addWidget(self.slider, 1)
        row_ctrl.addWidget(self.lbl_time)
        vbox.addLayout(row_ctrl)

        # Cut buttons
        row_cut = QHBoxLayout()
        self.btn_c1 = QPushButton("[ 1 ]  Corte → CAM 1")
        self.btn_c1.clicked.connect(lambda: self._add_cut(1))
        self.btn_c1.setStyleSheet(_btn("#12304a", C1_HEX, "#1a4060"))

        self.btn_c2 = QPushButton("[ 2 ]  Corte → CAM 2")
        self.btn_c2.clicked.connect(lambda: self._add_cut(2))
        self.btn_c2.setStyleSheet(_btn("#4a2e0a", C2_HEX, "#603a10"))

        self.btn_undo = QPushButton("↩  Deshacer  [ Z ]")
        self.btn_undo.clicked.connect(self._undo)
        self.btn_undo.setStyleSheet(_btn("#2a2a2a", "#555", "#383838"))

        self.lbl_cuts = QLabel("Cortes: 0")
        self.lbl_cuts.setStyleSheet("color:#888;font-size:12px;")

        row_cut.addWidget(self.btn_c1)
        row_cut.addWidget(self.btn_c2)
        row_cut.addWidget(self.btn_undo)
        row_cut.addStretch()
        row_cut.addWidget(self.lbl_cuts)
        vbox.addLayout(row_cut)

        # Section delete row
        row_sel = QHBoxLayout()
        row_sel.setSpacing(8)

        self.btn_select_mode = QPushButton("✂  Seleccionar sección  [ S ]")
        self.btn_select_mode.setCheckable(True)
        self.btn_select_mode.clicked.connect(self._toggle_select_mode)
        self.btn_select_mode.setStyleSheet(_btn("#2a1a00", "#a06000", "#3a2200"))

        self.btn_range_cam1 = QPushButton("→ CAM 1")
        self.btn_range_cam1.setEnabled(False)
        self.btn_range_cam1.clicked.connect(lambda: self._assign_range(1))
        self.btn_range_cam1.setStyleSheet(_btn("#12304a", C1_HEX, "#1a4060"))

        self.btn_range_cam2 = QPushButton("→ CAM 2")
        self.btn_range_cam2.setEnabled(False)
        self.btn_range_cam2.clicked.connect(lambda: self._assign_range(2))
        self.btn_range_cam2.setStyleSheet(_btn("#4a2e0a", C2_HEX, "#603a10"))

        self.btn_clear_cuts = QPushButton("✕  Limpiar cortes")
        self.btn_clear_cuts.setEnabled(False)
        self.btn_clear_cuts.clicked.connect(self._clear_range_cuts)
        self.btn_clear_cuts.setStyleSheet(_btn("#2a2a2a", "#777", "#383838"))

        self.btn_delete_sel = QPushButton("🗑  Borrar sección  [ Del ]")
        self.btn_delete_sel.setEnabled(False)
        self.btn_delete_sel.clicked.connect(self._delete_section)
        self.btn_delete_sel.setStyleSheet(_btn("#3a0000", "#aa2222", "#4a0000"))

        self.btn_undo_del = QPushButton("↩  Restaurar  [ Ctrl+Shift+Z ]")
        self.btn_undo_del.setEnabled(False)
        self.btn_undo_del.clicked.connect(self._undo_delete)
        self.btn_undo_del.setStyleSheet(_btn("#2a2a2a", "#555", "#383838"))

        self.lbl_sel_info = QLabel("Sin selección")
        self.lbl_sel_info.setStyleSheet("color:#888;font-size:11px;")

        row_sel.addWidget(self.btn_select_mode)
        row_sel.addWidget(self.btn_range_cam1)
        row_sel.addWidget(self.btn_range_cam2)
        row_sel.addWidget(self.btn_clear_cuts)
        row_sel.addWidget(self.btn_delete_sel)
        row_sel.addWidget(self.btn_undo_del)
        row_sel.addStretch()
        row_sel.addWidget(self.lbl_sel_info)
        vbox.addLayout(row_sel)

        # Auto-analyze row
        row_auto = QHBoxLayout()
        row_auto.setSpacing(8)

        lbl_auto = QLabel("Análisis automático:")
        lbl_auto.setStyleSheet("color:#aaa;font-size:12px;")

        self.combo_model = QComboBox()
        self.combo_model.addItems(["yolov8n  (rápido)", "yolov8s", "yolov8m  (preciso)"])
        self.combo_model.setStyleSheet(
            "QComboBox{background:#2a2a2a;color:#ddd;border:1px solid #444;"
            "border-radius:4px;padding:4px 8px;}"
            "QComboBox QAbstractItemView{background:#2a2a2a;color:#ddd;}")
        self.combo_model.setFixedWidth(180)

        self.combo_skip = QComboBox()
        self.combo_skip.addItems(["skip 2  (recomendado)", "skip 3  (más rápido)", "skip 1  (máx. calidad)"])
        self.combo_skip.setStyleSheet(self.combo_model.styleSheet())
        self.combo_skip.setFixedWidth(160)

        self.btn_auto = QPushButton("🔍  Analizar automáticamente")
        self.btn_auto.clicked.connect(self._auto_analyze)
        self.btn_auto.setStyleSheet(_btn("#1a1a3a", "#5060c0", "#22226a"))

        self.lbl_auto_status = QLabel("")
        self.lbl_auto_status.setStyleSheet("color:#888;font-size:11px;")

        row_auto.addWidget(lbl_auto)
        row_auto.addWidget(self.combo_model)
        row_auto.addWidget(self.combo_skip)
        row_auto.addWidget(self.btn_auto)
        row_auto.addWidget(self.lbl_auto_status, 1)
        vbox.addLayout(row_auto)

        # Action buttons
        row_act = QHBoxLayout()
        self.btn_open = QPushButton("📂  Abrir vídeos")
        self.btn_open.clicked.connect(self._open)
        self.btn_open.setStyleSheet(_btn("#2a2a2a", "#555", "#383838"))

        self.btn_preview = QPushButton("▶  Preview")
        self.btn_preview.setToolTip("Reproduce desde el inicio mostrando la cámara activa")
        self.btn_preview.clicked.connect(self._preview)
        self.btn_preview.setStyleSheet(_btn("#1a3a1a", "#3a7a3a", "#224422"))

        self.btn_gen = QPushButton("🎬  Generar vídeo")
        self.btn_gen.clicked.connect(self._generate)
        self.btn_gen.setStyleSheet(_btn("#30184a", "#7040a0", "#3e2060"))

        row_act.addWidget(self.btn_open)
        row_act.addStretch()
        row_act.addWidget(self.btn_preview)
        row_act.addWidget(self.btn_gen)
        vbox.addLayout(row_act)

    def _bind_shortcuts(self):
        for seq, fn in [
            ("Space",   self._toggle_play),
            ("1",       lambda: self._add_cut(1)),
            ("2",       lambda: self._add_cut(2)),
            ("Ctrl+Z",  self._undo),
            ("Left",    lambda: self._step(-1)),
            ("Right",   lambda: self._step(1)),
            ("Shift+Left",  lambda: self._step(-30)),
            ("Shift+Right",   lambda: self._step(30)),
            ("S",             self._toggle_select_mode_key),
            ("Delete",        self._delete_section),
            ("Ctrl+Shift+Z",  self._undo_delete),
        ]:
            QShortcut(QKeySequence(seq), self).activated.connect(fn)

    def _set_controls_enabled(self, on: bool):
        for w in (self.btn_play, self.btn_c1, self.btn_c2, self.btn_undo,
                  self.btn_preview, self.btn_gen, self.btn_auto,
                  self.btn_sync, self.slider):
            w.setEnabled(on)

    # ── loading ───────────────────────────────────────────────────────────────

    def _open(self):
        p1, _ = QFileDialog.getOpenFileName(
            self, "CAM 1", str(Path("videos")), "Vídeo (*.mp4 *.avi *.mov *.mkv)")
        if not p1: return
        p2, _ = QFileDialog.getOpenFileName(
            self, "CAM 2", str(Path("videos")), "Vídeo (*.mp4 *.avi *.mov *.mkv)")
        if not p2: return
        self._load(p1, p2)

    def _load(self, p1: str, p2: str):
        for c in (self.cap1, self.cap2):
            if c: c.release()
        self.cam1_path, self.cam2_path = p1, p2
        self.cap1 = cv2.VideoCapture(p1)
        self.cap2 = cv2.VideoCapture(p2)
        if not self.cap1.isOpened() or not self.cap2.isOpened():
            QMessageBox.critical(self, "Error", "No se pudieron abrir los vídeos.")
            return

        self.fps           = self.cap1.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames  = int(self.cap1.get(cv2.CAP_PROP_FRAME_COUNT))
        self.total_frames2 = int(self.cap2.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration      = min(self.total_frames, self.total_frames2) / self.fps
        self.cuts          = CutList()
        self.current_frame = 0
        # Reset sync y secciones borradas
        self.cam2_offset_sec    = 0.0
        self.cam2_offset_frames = 0
        self._last_cam2_frame   = -999
        self.deleted_ranges     = []
        self._deleted_history   = []
        self._current_sel       = None
        self.timeline.set_deleted([])
        self.lbl_offset.setText("Desfase: sin sincronizar  —  pulsa 🔊 para detectar")
        self.slider.setRange(0, int(self.duration * self.fps) - 1)
        self.timer.setInterval(max(1, int(1000 / self.fps)))
        self._set_controls_enabled(True)
        self._seek_frame(0)
        self.setWindowTitle(
            f"Basketball Editor — {Path(p1).name}  +  {Path(p2).name}")

    # ── playback ──────────────────────────────────────────────────────────────

    def _tick(self):
        ok1, f1 = self.cap1.read()
        if not ok1 or self.current_frame >= self.total_frames - 1:
            self._pause()
            return
        self.current_frame += 1

        # cam2 frame with offset — read sequentially when possible
        cam2_n = self.current_frame + self.cam2_offset_frames
        cam2_n = max(0, min(cam2_n, self.total_frames2 - 1))
        if cam2_n != self._last_cam2_frame + 1:
            self.cap2.set(cv2.CAP_PROP_POS_FRAMES, cam2_n)
        ok2, f2 = self.cap2.read()
        self._last_cam2_frame = cam2_n

        self._show(f1, f2)

    def _toggle_play(self):
        if self.cap1 is None: return
        self._pause() if self.playing else self._play()

    def _play(self):
        self.playing = True
        self.btn_play.setText("⏸")
        self.timer.start()

    def _pause(self):
        self.playing = False
        self.btn_play.setText("▶")
        self.timer.stop()

    def _step(self, delta: int):
        if self.cap1 is None: return
        self._pause()
        self._seek_frame(self.current_frame + delta)

    def _seek_sec(self, t: float):
        self._seek_frame(int(t * self.fps))

    def _seek_frame(self, n: int):
        if self.cap1 is None: return
        n = max(0, min(n, self.total_frames - 1))
        self.current_frame = n

        self.cap1.set(cv2.CAP_PROP_POS_FRAMES, n)
        ok1, f1 = self.cap1.read()

        cam2_n = n + self.cam2_offset_frames
        cam2_n = max(0, min(cam2_n, self.total_frames2 - 1))
        self.cap2.set(cv2.CAP_PROP_POS_FRAMES, cam2_n)
        ok2, f2 = self.cap2.read()
        self._last_cam2_frame = cam2_n

        self._show(f1, f2)

    def _slider_moved(self, v: int):
        self._pause()
        self._seek_frame(v)

    def _show(self, f1, f2):
        t      = self.current_frame / self.fps
        active = self.cuts.active_at(t)

        # Seek slider
        self.slider.blockSignals(True)
        self.slider.setValue(self.current_frame)
        self.slider.blockSignals(False)

        # Time
        self.lbl_time.setText(f"{fmt(t)} / {fmt(self.duration)}")

        # Frames
        self.pnl1.show_frame(f1, active == 1)
        self.pnl2.show_frame(f2, active == 2)

        # Timeline + cuts label
        self.timeline.set_state(self.duration, t, self.cuts.cuts)
        self.lbl_cuts.setText(f"Cortes: {self.cuts.n_cuts()}")

    # ── cuts ──────────────────────────────────────────────────────────────────

    def _add_cut(self, cam: int):
        if self.cap1 is None: return
        self.cuts.add(self.current_frame / self.fps, cam)
        t = self.current_frame / self.fps
        self.timeline.set_state(self.duration, t, self.cuts.cuts)
        self.lbl_cuts.setText(f"Cortes: {self.cuts.n_cuts()}")

    def _undo(self):
        self.cuts.undo()
        t = self.current_frame / self.fps
        self.timeline.set_state(self.duration, t, self.cuts.cuts)
        self.lbl_cuts.setText(f"Cortes: {self.cuts.n_cuts()}")

    # ── preview ───────────────────────────────────────────────────────────────

    def _preview(self):
        if self.cap1 is None: return
        self._seek_frame(0)
        self._play()

    # ── audio sync ───────────────────────────────────────────────────────────

    def _sync_audio(self):
        if self.cap1 is None:
            return
        self._pause()
        self._set_controls_enabled(False)
        self.lbl_offset.setText("Sincronizando…")

        self._sync_thread = SyncThread(self.cam1_path, self.cam2_path)
        self._sync_thread.status.connect(self.lbl_offset.setText)
        self._sync_thread.done.connect(self._on_sync_done)
        self._sync_thread.failed.connect(self._on_sync_failed)
        self._sync_thread.start()

    def _on_sync_done(self, offset_sec: float):
        self.cam2_offset_sec    = offset_sec
        self.cam2_offset_frames = round(offset_sec * self.fps)
        self._last_cam2_frame   = -999   # force re-seek on next display

        sign  = "+" if offset_sec >= 0 else ""
        early = "adelantada" if offset_sec >= 0 else "retrasada"
        self.lbl_offset.setText(
            f"Desfase: CAM 2 {sign}{offset_sec:.3f}s ({sign}{self.cam2_offset_frames} frames) "
            f"— CAM 2 está {early} respecto a CAM 1")
        self._set_controls_enabled(True)
        # Refresh current frame with new offset
        self._seek_frame(self.current_frame)

    def _on_sync_failed(self, err: str):
        self.lbl_offset.setText("✗ Error en sincronización — revisa que ffmpeg esté instalado")
        self._set_controls_enabled(True)
        QMessageBox.critical(self, "Error de sincronización",
                             f"No se pudo calcular el desfase:\n\n{err[-600:]}")

    # ── auto-analyze ─────────────────────────────────────────────────────────

    def _auto_analyze(self):
        if self.cap1 is None:
            return

        model_map = {"yolov8n  (rápido)": "yolov8n.pt",
                     "yolov8s":           "yolov8s.pt",
                     "yolov8m  (preciso)": "yolov8m.pt"}
        skip_map  = {"skip 2  (recomendado)": 2,
                     "skip 3  (más rápido)":  3,
                     "skip 1  (máx. calidad)": 1}
        model = model_map[self.combo_model.currentText()]
        skip  = skip_map[self.combo_skip.currentText()]

        self._pause()
        self._set_controls_enabled(False)
        self.lbl_auto_status.setText("Iniciando…")

        total = self.total_frames // skip
        fps   = self.fps

        # Estimate time (batch = 2 frames per call)
        fps_gpu = 30.0 / skip   # ~15 pairs/s × 2 frames on RTX 4090
        fps_cpu =  3.0 / skip   # conservative CPU estimate
        est_gpu = total / max(fps_gpu, 0.1) / 60
        est_cpu = total / max(fps_cpu, 0.1) / 60
        self.lbl_auto_status.setText(
            f"Estimado: ~{est_gpu:.0f} min (GPU)  /  ~{est_cpu:.0f} min (CPU)")

        self._prog_auto = QProgressDialog(
            "Analizando vídeos…", "Cancelar", 0, total, self)
        self._prog_auto.setWindowTitle("Análisis automático")
        self._prog_auto.setWindowModality(Qt.WindowModality.WindowModal)
        self._prog_auto.setMinimumDuration(0)
        self._prog_auto.setMinimumWidth(480)
        self._prog_auto.setValue(0)

        self._auto_thread = AutoAnalyzeThread(
            self.cam1_path, self.cam2_path,
            fps, self.total_frames, model, skip)
        self._auto_thread.progress.connect(self._on_auto_progress)
        self._auto_thread.done.connect(self._on_auto_done)
        self._auto_thread.failed.connect(self._on_auto_failed)
        self._prog_auto.canceled.connect(self._auto_thread.abort)
        self._prog_auto.canceled.connect(
            lambda: self._set_controls_enabled(True))
        self._auto_thread.start()

    def _on_auto_progress(self, current, total, msg):
        if self._prog_auto:
            self._prog_auto.setValue(current)
            self._prog_auto.setLabelText(msg)
        self.lbl_auto_status.setText(msg)

    def _on_auto_done(self, cuts):
        self._prog_auto.close()
        # Load cuts into model
        self.cuts = CutList()
        for t, cam in cuts:
            if t == 0.0:
                self.cuts._cuts[0] = (0.0, cam)
            else:
                bisect.insort(self.cuts._cuts, (t, cam))
        self.cuts._collapse()

        n = self.cuts.n_cuts()
        self.lbl_auto_status.setText(
            f"✓ Análisis completo — {n} corte{'s' if n != 1 else ''} detectado{'s' if n != 1 else ''}. "
            f"Revisa y ajusta manualmente si lo necesitas.")
        self._set_controls_enabled(True)
        # Refresh display
        t = self.current_frame / self.fps
        self.timeline.set_state(self.duration, t, self.cuts.cuts)
        self.lbl_cuts.setText(f"Cortes: {n}")

    def _on_auto_failed(self, err):
        self._prog_auto.close()
        self._set_controls_enabled(True)
        self.lbl_auto_status.setText("✗ Error en el análisis.")
        QMessageBox.critical(self, "Error en análisis", err)

    # ── generate ──────────────────────────────────────────────────────────────

    # ── section delete ────────────────────────────────────────────────────────

    def _toggle_select_mode_key(self):
        self.btn_select_mode.setChecked(not self.btn_select_mode.isChecked())
        self._toggle_select_mode(self.btn_select_mode.isChecked())

    def _toggle_select_mode(self, checked: bool):
        self.timeline.set_select_mode(checked)
        if checked:
            self.btn_select_mode.setText("✂  Modo selección: ON  [ S ]")
            self._current_sel = None
            self._set_sel_buttons(False)
            self.lbl_sel_info.setText("Arrastra sobre la línea de tiempo para seleccionar")
        else:
            self.btn_select_mode.setText("✂  Seleccionar sección  [ S ]")
            self.timeline.clear_selection()
            self._current_sel = None
            self._set_sel_buttons(False)
            self.lbl_sel_info.setText("Sin selección")

    def _set_sel_buttons(self, on: bool):
        for b in (self.btn_range_cam1, self.btn_range_cam2,
                  self.btn_clear_cuts, self.btn_delete_sel):
            b.setEnabled(on)

    def _on_section_selected(self, start: float, end: float):
        self._current_sel = (start, end)
        self._set_sel_buttons(True)
        self.lbl_sel_info.setText(
            f"Selección: {fmt(start)} → {fmt(end)}  ({end - start:.1f}s)")

    def _assign_range(self, cam: int):
        """Asigna toda la selección a cam 1 o 2."""
        if not self._current_sel:
            return
        a, b = self._current_sel
        self.cuts.set_range(a, b, cam)
        self._current_sel = None
        self._finish_sel_action()
        self.lbl_sel_info.setText(
            f"Sección {fmt(a)}–{fmt(b)} asignada a CAM {cam}")

    def _clear_range_cuts(self):
        """Elimina los cortes internos de la selección."""
        if not self._current_sel:
            return
        a, b = self._current_sel
        self.cuts.clear_range(a, b)
        self._current_sel = None
        self._finish_sel_action()
        self.lbl_sel_info.setText(
            f"Cortes eliminados en {fmt(a)}–{fmt(b)}")

    def _finish_sel_action(self):
        """Refresca UI y desactiva modo selección tras una acción."""
        t = self.current_frame / self.fps
        self.timeline.set_state(self.duration, t, self.cuts.cuts)
        self.lbl_cuts.setText(f"Cortes: {self.cuts.n_cuts()}")
        self.btn_select_mode.setChecked(False)
        self._toggle_select_mode(False)

    def _delete_section(self):
        if not self._current_sel:
            return
        self._deleted_history.append(list(self.deleted_ranges))
        a, b = self._current_sel
        self.deleted_ranges.append((a, b))
        self.deleted_ranges = _merge_ranges(self.deleted_ranges)
        self._current_sel = None
        self.btn_delete_sel.setEnabled(False)
        self.btn_undo_del.setEnabled(True)
        n = len(self.deleted_ranges)
        total_del = sum(e - s for s, e in self.deleted_ranges)
        self.lbl_sel_info.setText(
            f"{n} sección(s) borrada(s) — {total_del:.1f}s eliminados del export")
        self.timeline.set_deleted(self.deleted_ranges)
        self.timeline.clear_selection()
        # Desactivar modo selección
        self.btn_select_mode.setChecked(False)
        self._toggle_select_mode(False)

    def _undo_delete(self):
        if self._deleted_history:
            self.deleted_ranges = self._deleted_history.pop()
            self.timeline.set_deleted(self.deleted_ranges)
            self.btn_undo_del.setEnabled(bool(self._deleted_history))
            if self.deleted_ranges:
                self.lbl_sel_info.setText(
                    f"{len(self.deleted_ranges)} sección(s) borrada(s)")
            else:
                self.lbl_sel_info.setText("Sin selección")

    # ── generate ──────────────────────────────────────────────────────────────

    def _generate(self):
        if self.cap1 is None: return
        out, _ = QFileDialog.getSaveFileName(
            self, "Guardar vídeo", "output_edited.mp4", "Vídeo (*.mp4)")
        if not out: return
        self._pause()

        self._prog = QProgressDialog("Generando vídeo…", "Cancelar", 0, 100, self)
        self._prog.setWindowModality(Qt.WindowModality.WindowModal)
        self._prog.setMinimumDuration(0)
        self._prog.setValue(0)

        self._gen = GeneratorThread(
            self.cam1_path, self.cam2_path, out,
            self.cuts.cuts, self.fps, self.total_frames,
            cam2_offset_frames=self.cam2_offset_frames,
            total_frames2=self.total_frames2,
            deleted_ranges=self.deleted_ranges)
        self._gen.progress.connect(
            lambda n, t: self._prog.setValue(int(100 * n / t)))
        self._gen.done.connect(self._gen_done)
        self._gen.failed.connect(self._gen_failed)
        self._prog.canceled.connect(self._gen.abort)
        self._gen.start()

    def _gen_done(self, path):
        self._prog.close()
        QMessageBox.information(self, "✓ Listo", f"Vídeo guardado en:\n{path}")

    def _gen_failed(self, err):
        self._prog.close()
        QMessageBox.critical(self, "Error", err)

    # ── cleanup ───────────────────────────────────────────────────────────────

    def closeEvent(self, _ev):
        self.timer.stop()
        if self.cap1: self.cap1.release()
        if self.cap2: self.cap2.release()


# ── video panel widget ────────────────────────────────────────────────────────

class _VideoPanel(QFrame):
    def __init__(self, title: str, accent: str):
        super().__init__()
        self._accent = accent
        self._set_border(False)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        self._header = QLabel(title)
        self._header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._header.setStyleSheet(
            f"background:{accent}22;color:{accent};font-weight:bold;"
            f"padding:5px;border-bottom:1px solid #333;")
        layout.addWidget(self._header)

        # Video label
        self._lbl = QLabel("Sin vídeo")
        self._lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl.setStyleSheet("color:#444;font-size:13px;")
        self._lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self._lbl, 1)

        # Active badge
        self._badge = QLabel("◀  ACTIVA")
        self._badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._badge.setStyleSheet(
            f"background:{accent};color:black;font-weight:bold;"
            f"padding:4px;font-size:12px;")
        self._badge.setVisible(False)
        layout.addWidget(self._badge)

    def _set_border(self, active: bool):
        col = self._accent if active else "#2e2e2e"
        w   = 3 if active else 2
        self.setStyleSheet(
            f"_VideoPanel{{background:{PANEL};border:{w}px solid {col};"
            f"border-radius:6px;}}")

    def show_frame(self, frame, active: bool):
        if frame is not None:
            self._lbl.setPixmap(to_pixmap(frame, self._lbl.size()))
        self._set_border(active)
        self._badge.setVisible(active)


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    p1 = sys.argv[1] if len(sys.argv) > 1 else None
    p2 = sys.argv[2] if len(sys.argv) > 2 else None

    win = BasketballEditor(p1, p2)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
