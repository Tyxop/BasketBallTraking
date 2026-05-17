#!/usr/bin/env bash
# Abre el editor GUI.
# Uso:
#   ./gui.sh
#   ./gui.sh videos/cam1.mp4 videos/cam2.mp4

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

conda run --no-capture-output -n basketball python -u gui.py "$@"
