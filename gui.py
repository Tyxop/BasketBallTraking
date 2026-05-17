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


# ── timeline widget ───────────────────────────────────────────────────────────

class TimelineWidget(QWidget):
    seek_requested = pyqtSignal(float)

    def __init__(self):
        super().__init__()
        self.duration = 1.0
        self.position = 0.0
        self.cuts: list[tuple[float, int]] = [(0.0, 1)]
        self.setMinimumHeight(72)
        self.setMaximumHeight(88)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_state(self, duration, position, cuts):
        self.duration = max(duration, 1.0)
        self.position = position
        self.cuts     = cuts
        self.update()

    def _x(self, t: float) -> int:
        return int(t / self.duration * (self.width() - 1))

    def paintEvent(self, _ev):
        p  = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        BY, BH = 16, 32       # bar Y offset and height

        # Background
        p.fillRect(0, 0, W, H, QColor(20, 20, 20))

        # Colored segments
        for i, (ct, cam) in enumerate(self.cuts):
            nt   = self.cuts[i + 1][0] if i + 1 < len(self.cuts) else self.duration
            x1   = self._x(ct)
            x2   = self._x(nt)
            col  = C1_COL if cam == 1 else C2_COL
            p.fillRect(x1, BY, max(x2 - x1, 1), BH, col)

        # Cut-point markers (skip the t=0 one)
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

        # Playhead line
        px = self._x(self.position)
        p.setPen(QPen(QColor(255, 255, 255), 2))
        p.drawLine(px, BY - 8, px, BY + BH + 8)

        # Playhead triangle at the top
        p.setBrush(QBrush(QColor(255, 255, 255)))
        p.setPen(Qt.PenStyle.NoPen)
        tri = QPolygon([
            QPoint(px,     BY + 6),
            QPoint(px - 6, BY - 2),
            QPoint(px + 6, BY - 2),
        ])
        p.drawPolygon(tri)

        # Legend (top-right)
        p.setFont(QFont("sans-serif", 8))
        for i, (lbl, col) in enumerate([("CAM 1", C1_COL), ("CAM 2", C2_COL)]):
            lx = W - 130 + i * 65
            p.fillRect(lx, 3, 12, 10, col)
            p.setPen(QPen(QColor(200, 200, 200)))
            p.drawText(lx + 14, 12, lbl)

        p.end()

    def mousePressEvent(self, ev):
        self._emit_seek(ev)

    def mouseMoveEvent(self, ev):
        if ev.buttons() & Qt.MouseButton.LeftButton:
            self._emit_seek(ev)

    def _emit_seek(self, ev):
        t = ev.position().x() / self.width() * self.duration
        self.seek_requested.emit(max(0.0, min(t, self.duration)))


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
    def _score_frame(frame, model, tracker, states):
        """Returns (score, tracker_states_updated)."""
        from collections import deque
        h = frame.shape[0]
        results = model(frame, classes=[0, 32], conf=0.25, verbose=False)[0]

        person_dets, yolo_balls = [], []
        for box in results.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            if cls == 32:
                yolo_balls.append([x1, y1, x2, y2])
            elif cls == 0 and conf >= 0.35:
                person_dets.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

        ball = (yolo_balls or AutoAnalyzeThread._ball_color(frame))
        ball = ball[0] if ball else None

        tracks   = tracker.update_tracks(person_dets, frame=frame)
        moving   = 0
        sitting  = 0
        has_ball = False

        for track in tracks:
            if not track.is_confirmed():
                continue
            tid = track.track_id
            x1, y1, x2, y2 = track.to_ltrb()
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
                # proximity check
                bx, by = (ball[0]+ball[2])/2, (ball[1]+ball[3])/2
                px, py = (x1+x2)/2, (y1+y2)/2
                if ((bx-px)**2 + (by-py)**2)**0.5 / max(h, 1) < 0.12:
                    has_ball = True

        score = 0
        if ball:        score += 200
        if has_ball:    score += 300
        score += moving  * 30
        score += sitting * 5
        return score

    def run(self):
        try:
            self.progress.emit(0, 1, "Cargando modelo…")
            from ultralytics import YOLO
            from deep_sort_realtime.deepsort_tracker import DeepSort

            model   = YOLO(self.model_name)
            track1  = DeepSort(max_age=30, n_init=3, nn_budget=100)
            track2  = DeepSort(max_age=30, n_init=3, nn_budget=100)
            states1: dict = {}
            states2: dict = {}

            cap1  = cv2.VideoCapture(self.cam1)
            cap2  = cv2.VideoCapture(self.cam2)
            total = self.total // self.skip
            raw   = []
            t0    = time.time()

            self.progress.emit(0, total, "Analizando…")

            idx = 0
            while idx * self.skip < self.total:
                if self._abort:
                    cap1.release(); cap2.release()
                    return

                for _ in range(self.skip - 1):
                    cap1.grab(); cap2.grab()

                ok1, f1 = cap1.read()
                ok2, f2 = cap2.read()
                if not ok1 or not ok2:
                    break

                s1 = self._score_frame(f1, model, track1, states1)
                s2 = self._score_frame(f2, model, track2, states2)
                raw.append((s1, s2))
                idx += 1

                if idx % 15 == 0:
                    elapsed  = time.time() - t0
                    fps_real = idx / max(elapsed, 0.01)
                    eta      = (total - idx) / max(fps_real, 0.01)
                    self.progress.emit(
                        idx, total,
                        f"Analizando… {idx}/{total}  "
                        f"{fps_real:.1f} fr/s  ETA {eta/60:.1f} min")

            cap1.release(); cap2.release()

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

    def __init__(self, cam1, cam2, output, cuts, fps, total):
        super().__init__()
        self.cam1, self.cam2 = cam1, cam2
        self.output, self.cuts, self.fps, self.total = output, cuts, fps, total
        self._abort = False

    def abort(self):
        self._abort = True

    def _active_at(self, t):
        cam = self.cuts[0][1]
        for ct, c in self.cuts:
            if ct <= t: cam = c
            else: break
        return cam

    def run(self):
        try:
            cap1 = cv2.VideoCapture(self.cam1)
            cap2 = cv2.VideoCapture(self.cam2)
            W  = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
            H  = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
            W2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
            H2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(
                self.output, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (W, H))

            for n in range(self.total):
                if self._abort:
                    break
                ok1, f1 = cap1.read()
                ok2, f2 = cap2.read()
                active = self._active_at(n / self.fps)
                frame  = f1 if active == 1 else f2
                if frame is not None:
                    # Resize cam2 frames if resolution differs
                    if active == 2 and (W2 != W or H2 != H):
                        frame = cv2.resize(frame, (W, H))
                    writer.write(frame)
                if n % 120 == 0:
                    self.progress.emit(n, self.total)

            cap1.release(); cap2.release(); writer.release()
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
        self.total_frames = 0
        self.duration = 0.0
        self.current_frame = 0
        self.playing = False
        self.cuts = CutList()
        self._gen: GeneratorThread | None = None

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

        # Timeline
        self.timeline = TimelineWidget()
        self.timeline.seek_requested.connect(self._seek_sec)
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
            ("Shift+Right", lambda: self._step(30)),
        ]:
            QShortcut(QKeySequence(seq), self).activated.connect(fn)

    def _set_controls_enabled(self, on: bool):
        for w in (self.btn_play, self.btn_c1, self.btn_c2, self.btn_undo,
                  self.btn_preview, self.btn_gen, self.btn_auto, self.slider):
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

        self.fps          = self.cap1.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = min(int(self.cap1.get(cv2.CAP_PROP_FRAME_COUNT)),
                                int(self.cap2.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.duration     = self.total_frames / self.fps
        self.cuts         = CutList()
        self.current_frame = 0
        self.slider.setRange(0, self.total_frames - 1)
        self.timer.setInterval(max(1, int(1000 / self.fps)))
        self._set_controls_enabled(True)
        self._seek_frame(0)
        self.setWindowTitle(
            f"Basketball Editor — {Path(p1).name}  +  {Path(p2).name}")

    # ── playback ──────────────────────────────────────────────────────────────

    def _tick(self):
        ok1, f1 = self.cap1.read()
        ok2, f2 = self.cap2.read()
        if not ok1 or not ok2 or self.current_frame >= self.total_frames - 1:
            self._pause()
            return
        self.current_frame += 1
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
        self.cap2.set(cv2.CAP_PROP_POS_FRAMES, n)
        ok1, f1 = self.cap1.read()
        ok2, f2 = self.cap2.read()
        # caps are now at n+1; next tick reads n+1 correctly
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

        # Estimate time
        fps_cpu = 3.0 / skip   # conservative CPU estimate
        est_min = total / max(fps_cpu, 0.1) / 60
        self.lbl_auto_status.setText(
            f"Tiempo estimado: ~{est_min:.0f} min (CPU).  "
            f"Con GPU puede ser mucho más rápido.")

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
            self.cuts.cuts, self.fps, self.total_frames)
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
