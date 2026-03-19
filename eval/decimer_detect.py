#!/usr/bin/env python3
"""DECIMER segmentation helper — run in venv_decimer.

Batch mode: reads a JSON manifest of image paths, runs DECIMER on each,
outputs JSON results. Model loads once for all images.

Usage:
    venv_decimer/bin/python eval/decimer_detect.py <manifest.json>
    # manifest: [{"path": "/tmp/img.png", "key": "GLP_98"}, ...]
    # outputs JSON: [{"key": "GLP_98", "count": N, "boxes": [...], "time": T}, ...]
"""

import json
import sys
import time

import numpy as np
from PIL import Image

from decimer_segmentation import segment_chemical_structures


def main():
    manifest_path = sys.argv[1]
    with open(manifest_path) as f:
        manifest = json.load(f)

    results = []
    for entry in manifest:
        image_path = entry["path"]
        key = entry["key"]

        pil_image = Image.open(image_path).convert("RGB")
        img_array = np.array(pil_image)

        t0 = time.time()
        segments, bboxes = segment_chemical_structures(
            img_array, expand=True, visualization=False, return_bboxes=True
        )
        elapsed = time.time() - t0

        results.append({
            "key": key,
            "count": len(bboxes),
            "boxes": [list(map(int, b)) for b in bboxes],
            "time": round(elapsed, 3),
        })

        # Print progress to stderr
        print(f"  DECIMER: {key} -> {len(bboxes)} structures ({elapsed:.2f}s)", file=sys.stderr)

    print(json.dumps(results))


if __name__ == "__main__":
    main()
