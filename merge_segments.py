"""
Merges two or more segment JSONs produced by analyze.py (--start/--end)
into a single JSON for render.py.

Usage:
  python merge_segments.py --inputs cam1_a.json cam1_b.json --output cam1.json
"""

import json
import argparse
from pathlib import Path


def merge(inputs: list[str], output: str):
    merged_frames = []
    meta = None

    for path in inputs:
        data = json.loads(Path(path).read_text())
        if meta is None:
            meta = {k: v for k, v in data.items() if k != 'frames'}
        merged_frames.extend(data['frames'])

    # Sort by frame index (segments may arrive out of order)
    merged_frames.sort(key=lambda f: f['frame'])

    result = {**meta, 'frames': merged_frames}
    Path(output).write_text(json.dumps(result, separators=(',', ':')))
    print(f"Merged {len(merged_frames)} frames → {output}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    merge(args.inputs, args.output)
