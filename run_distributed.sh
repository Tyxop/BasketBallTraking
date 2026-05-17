#!/usr/bin/env bash
# ─── Distributed run — instructions ──────────────────────────────────────────
#
# Copia los videos y los scripts a cada máquina, ejecuta analyze.py en paralelo
# y luego render.py cuando ambos terminen.
#
# REQUISITOS en cada máquina:
#   pip install ultralytics deep_sort_realtime opencv-python tqdm "numpy<2" "setuptools<71"
#   (en Windows con CUDA: instala primero PyTorch con CUDA antes de lo anterior)
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
#
# OPCIÓN A — Split por cámara (2 máquinas)
# ─────────────────────────────────────────
#   Máquina A (ej. 4090):
#     python analyze.py --cam cam1.mp4 --output cam1.json --model yolov8m.pt --skip 1
#
#   Máquina B (ej. 5090 Mobile):
#     python analyze.py --cam cam2.mp4 --output cam2.json --model yolov8m.pt --skip 1
#
#   Cuando ambas terminen — copiar cam1.json y cam2.json a una máquina y:
#     python render.py --cam1 cam1.mp4 --cam2 cam2.mp4 --json1 cam1.json --json2 cam2.json
#
#
# OPCIÓN B — Split por cámara Y por segmento (4 máquinas)
# ─────────────────────────────────────────────────────────
#   Divide cada video en dos mitades (~11600 frames):
#
#   Máquina A (4090)        → cam1, frames 0–11600
#     python analyze.py --cam cam1.mp4 --output cam1_a.json --start 0 --end 11600
#
#   Máquina B (5090 Mobile) → cam1, frames 11600–EOF
#     python analyze.py --cam cam1.mp4 --output cam1_b.json --start 11600
#
#   Máquina C (M5)          → cam2, frames 0–11600
#     python analyze.py --cam cam2.mp4 --output cam2_a.json --start 0 --end 11600
#
#   Máquina D (M4)          → cam2, frames 11600–EOF
#     python analyze.py --cam cam2.mp4 --output cam2_b.json --start 11600
#
#   Luego fusionar segmentos con merge_segments.py y renderizar:
#     python merge_segments.py --inputs cam1_a.json cam1_b.json --output cam1.json
#     python merge_segments.py --inputs cam2_a.json cam2_b.json --output cam2.json
#     python render.py --cam1 cam1.mp4 --cam2 cam2.mp4 --json1 cam1.json --json2 cam2.json
#
#
# VELOCIDAD ESTIMADA (yolov8m, skip=1):
#   RTX 4090       →  ~50 fr/s  →  ~8 min por cámara
#   RTX 5090 Mobile→  ~40 fr/s  →  ~10 min por cámara
#   M4/M5 (MPS)    →  ~20 fr/s  →  ~20 min por cámara
#   CPU x86        →  ~3 fr/s   →  ~130 min por cámara
#
# Con Opción A (cam split):        tiempo total ≈ max(cam1, cam2) ≈ 8 min
# Con Opción B (cam + seg split):  tiempo total ≈ 4–5 min
# ─────────────────────────────────────────────────────────────────────────────

echo "Este script es solo documentación. Lee los comentarios internos."
echo "Ejecuta analyze.py y render.py manualmente en cada máquina."
