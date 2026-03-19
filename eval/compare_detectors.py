#!/usr/bin/env python3
"""Compare Docling vs DECIMER segmentation for structure detection.

Runs both detectors on all ground-truth pages, compares counts and timing.
DECIMER runs in a separate venv (venv_decimer) via subprocess due to TF version conflicts.

Usage:
    ./venv/bin/python eval/compare_detectors.py                    # all patents
    ./venv/bin/python eval/compare_detectors.py --patent GLP       # single patent
    ./venv/bin/python eval/compare_detectors.py --patent Frontier --pages 105 107
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pypdfium2 as pdfium
from PIL import Image, ImageDraw

from pdf_to_smiles.core.docling_classifier import DoclingClassifier
from pdf_to_smiles.core.docling_detector import DoclingDetector

PDF_DIR = os.path.expanduser("~/Downloads/Sample patents for testing")
DPI_SCALE = 200 / 72
GROUND_TRUTH_PATH = os.path.join(os.path.dirname(__file__), "ground_truth.json")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output", "compare")

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
DECIMER_PYTHON = os.path.join(PROJECT_ROOT, "venv_decimer", "bin", "python")
DECIMER_SCRIPT = os.path.join(os.path.dirname(__file__), "decimer_detect.py")


def load_ground_truth():
    with open(GROUND_TRUTH_PATH) as f:
        return json.load(f)


def run_decimer_batch(image_paths_and_keys):
    """Run DECIMER on a batch of images via subprocess.

    Args:
        image_paths_and_keys: list of (image_path, key_string) tuples

    Returns:
        dict mapping key -> {"count": N, "boxes": [...], "time": T}
    """
    manifest = [{"path": p, "key": k} for p, k in image_paths_and_keys]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(manifest, f)
        manifest_path = f.name

    try:
        result = subprocess.run(
            [DECIMER_PYTHON, DECIMER_SCRIPT, manifest_path],
            capture_output=True, text=True, timeout=3600,
        )
        if result.returncode != 0:
            print(f"DECIMER batch error: {result.stderr[-500:]}")
            return {}

        # Print DECIMER progress from stderr
        if result.stderr:
            for line in result.stderr.strip().split("\n"):
                print(line)

        results = json.loads(result.stdout.strip())
        return {r["key"]: r for r in results}
    finally:
        os.unlink(manifest_path)


def save_annotated(pil_image, boxes, label, color, path):
    """Save annotated image with bounding boxes."""
    annotated = pil_image.copy()
    draw = ImageDraw.Draw(annotated)
    for i, box in enumerate(boxes):
        if isinstance(box, (list, tuple)) and len(box) == 2 and not isinstance(box[0], (int, float)):
            _, box = box  # (crop, bbox) tuple from docling
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1 + 4, y1 + 4), f"[{i}]", fill=color)
    draw.rectangle([(0, 0), (pil_image.width, 28)], fill=(40, 40, 40))
    draw.text((8, 6), f"{label}: {len(boxes)} detected", fill=color)
    annotated.save(path)


def run_comparison(gt, patent_filter=None, page_filter=None):
    """Run full comparison across all selected patents and pages."""
    all_results = []

    # Phase 1: Render all pages and run Docling, save temp images for DECIMER
    temp_dir = tempfile.mkdtemp(prefix="decimer_eval_")
    decimer_manifest = []  # (temp_image_path, key)
    docling_results = {}   # key -> {"count", "time", "dets"}
    page_images = {}       # key -> pil_image
    page_configs = {}      # key -> page_cfg dict

    for patent_cfg in gt["patents"]:
        patent_id = patent_cfg["id"]
        if patent_filter and patent_id != patent_filter:
            continue

        pdf_path = os.path.join(PDF_DIR, patent_cfg["filename"])
        if not os.path.exists(pdf_path):
            print(f"  SKIP: {pdf_path} not found")
            continue

        patent_out = os.path.join(OUTPUT_DIR, patent_id)
        os.makedirs(patent_out, exist_ok=True)

        print(f"\n{'=' * 50}")
        print(f"Patent: {patent_id}")
        print(f"{'=' * 50}")

        # Init Docling
        print(f"  Initializing Docling classifier...")
        t0 = time.time()
        classifier = DoclingClassifier()
        classifier.classify_pdf(pdf_path)
        detector = DoclingDetector(classifier)
        docling_init_time = time.time() - t0
        print(f"  Docling init: {docling_init_time:.1f}s")

        doc = pdfium.PdfDocument(pdf_path)
        pages = patent_cfg["pages"]
        if page_filter:
            pages = [p for p in pages if p["page"] in page_filter]

        for page_cfg in pages:
            page_num = page_cfg["page"]
            if page_num > len(doc):
                print(f"  Page {page_num} exceeds document length, skipping")
                continue

            key = f"{patent_id}_{page_num}"

            # Render page
            page = doc[page_num - 1]
            bitmap = page.render(scale=DPI_SCALE)
            pil_image = bitmap.to_pil()
            page_images[key] = pil_image
            page_configs[key] = {**page_cfg, "patent": patent_id}

            # Run Docling
            t0 = time.time()
            dets = detector.detect_structures_with_boxes(pil_image, page_num)
            docling_time = time.time() - t0
            docling_results[key] = {
                "count": len(dets), "time": docling_time, "dets": dets,
            }
            print(f"  Docling p{page_num}: {len(dets)} structures ({docling_time:.2f}s)")

            # Save temp image for DECIMER batch
            tmp_path = os.path.join(temp_dir, f"{key}.png")
            pil_image.save(tmp_path)
            decimer_manifest.append((tmp_path, key))

    # Phase 2: Run DECIMER batch
    print(f"\n{'=' * 50}")
    print(f"Running DECIMER batch ({len(decimer_manifest)} pages)...")
    print(f"{'=' * 50}")

    t0 = time.time()
    decimer_results = run_decimer_batch(decimer_manifest)
    decimer_wall = time.time() - t0
    print(f"  DECIMER batch wall time: {decimer_wall:.1f}s")

    # Phase 3: Compile results
    for key in docling_results:
        cfg = page_configs[key]
        patent_id = cfg["patent"]
        page_num = cfg["page"]
        expected = cfg["expected_count"]
        pil_image = page_images[key]

        doc_r = docling_results[key]
        dec_r = decimer_results.get(key, {"count": 0, "boxes": [], "time": 0})

        result = {
            "patent": patent_id,
            "page": page_num,
            "expected": expected,
            "docling_count": doc_r["count"],
            "docling_delta": doc_r["count"] - expected,
            "docling_time": round(doc_r["time"], 2),
            "decimer_count": dec_r["count"],
            "decimer_delta": dec_r["count"] - expected,
            "decimer_time": round(dec_r["time"], 2),
            "notes": cfg.get("notes", ""),
        }
        all_results.append(result)

        # Save annotated images
        patent_out = os.path.join(OUTPUT_DIR, patent_id)
        save_annotated(
            pil_image, doc_r["dets"], f"Docling p{page_num}",
            (0, 200, 0) if result["docling_delta"] == 0 else (255, 0, 0),
            os.path.join(patent_out, f"page_{page_num:03d}_docling.png"),
        )
        save_annotated(
            pil_image, dec_r.get("boxes", []), f"DECIMER p{page_num}",
            (0, 100, 255) if result["decimer_delta"] == 0 else (255, 100, 0),
            os.path.join(patent_out, f"page_{page_num:03d}_decimer.png"),
        )

    # Cleanup temp files
    for tmp_path, _ in decimer_manifest:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    os.rmdir(temp_dir)

    return all_results


def print_scorecard(all_results):
    if not all_results:
        print("\nNo results.")
        return

    print("\n" + "=" * 90)
    print("COMPARISON SCORECARD: Docling vs DECIMER Segmentation")
    print("=" * 90)

    docling_pass = sum(1 for r in all_results if r["docling_delta"] == 0)
    decimer_pass = sum(1 for r in all_results if r["decimer_delta"] == 0)
    total = len(all_results)

    docling_total_time = sum(r["docling_time"] for r in all_results)
    decimer_total_time = sum(r["decimer_time"] for r in all_results)

    docling_total_exp = sum(r["expected"] for r in all_results)
    docling_total_det = sum(r["docling_count"] for r in all_results)
    decimer_total_det = sum(r["decimer_count"] for r in all_results)

    docling_abs_err = sum(abs(r["docling_delta"]) for r in all_results)
    decimer_abs_err = sum(abs(r["decimer_delta"]) for r in all_results)

    # Per-patent breakdown
    patents = {}
    for r in all_results:
        patents.setdefault(r["patent"], []).append(r)

    print(f"\n{'Patent':<10} {'Pages':>5}  {'Docling':>18}  {'DECIMER':>18}")
    print(f"{'':10} {'':>5}  {'pass/total  time':>18}  {'pass/total  time':>18}")
    print("-" * 65)

    for pid, res in patents.items():
        dp = sum(1 for r in res if r["docling_delta"] == 0)
        rp = sum(1 for r in res if r["decimer_delta"] == 0)
        dt = sum(r["docling_time"] for r in res)
        rt = sum(r["decimer_time"] for r in res)
        n = len(res)
        print(
            f"{pid:<10} {n:>5}  "
            f"{dp:>3d}/{n:<3d}  {dt:>7.1f}s  "
            f"{rp:>3d}/{n:<3d}  {rt:>7.1f}s"
        )

    print("-" * 65)
    print(
        f"{'TOTAL':<10} {total:>5}  "
        f"{docling_pass:>3d}/{total:<3d}  {docling_total_time:>7.1f}s  "
        f"{decimer_pass:>3d}/{total:<3d}  {decimer_total_time:>7.1f}s"
    )

    print(f"\n{'Metric':<30} {'Docling':>10} {'DECIMER':>10}")
    print("-" * 52)
    print(f"{'Exact match (pages)':30} {docling_pass:>8d}/{total} {decimer_pass:>8d}/{total}")
    dp = docling_pass / total * 100 if total else 0
    rp = decimer_pass / total * 100 if total else 0
    print(f"{'Accuracy (%)':30} {dp:>9.1f}% {rp:>9.1f}%")
    print(f"{'Total expected structures':30} {docling_total_exp:>10d} {docling_total_exp:>10d}")
    print(f"{'Total detected structures':30} {docling_total_det:>10d} {decimer_total_det:>10d}")
    print(f"{'Sum |delta| (abs error)':30} {docling_abs_err:>10d} {decimer_abs_err:>10d}")
    print(f"{'Total detection time':30} {docling_total_time:>9.1f}s {decimer_total_time:>9.1f}s")
    avg_d = docling_total_time / total if total else 0
    avg_r = decimer_total_time / total if total else 0
    print(f"{'Avg time per page':30} {avg_d:>9.2f}s {avg_r:>9.2f}s")

    # Per-page detail table
    print(f"\n{'Patent':<8} {'Page':>4} {'Exp':>4} {'Docl':>5} {'dD':>4} {'DECI':>5} {'dR':>4}  {'Notes'}")
    print("-" * 80)
    for r in all_results:
        d_mark = " " if r["docling_delta"] == 0 else "*"
        r_mark = " " if r["decimer_delta"] == 0 else "*"
        print(
            f"{r['patent']:<8} {r['page']:>4d} {r['expected']:>4d} "
            f"{r['docling_count']:>4d}{d_mark} {r['decimer_count']:>4d}{r_mark}  "
            f"{r.get('notes', '')}"
        )

    print("=" * 90)

    # Save JSON
    results_path = os.path.join(OUTPUT_DIR, "comparison_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    print(f"Annotated images saved to {OUTPUT_DIR}/")


def main():
    parser = argparse.ArgumentParser(description="Compare Docling vs DECIMER segmentation")
    parser.add_argument("--patent", type=str, help="Run only this patent ID")
    parser.add_argument("--pages", type=int, nargs="+", help="Run only these pages")
    args = parser.parse_args()

    if not os.path.exists(DECIMER_PYTHON):
        print(f"ERROR: DECIMER venv not found at {DECIMER_PYTHON}")
        print("Create it: python3.11 -m venv venv_decimer && venv_decimer/bin/pip install 'decimer-segmentation>=1.4.0' 'tensorflow<=2.15.1' pypdfium2 pillow")
        sys.exit(1)

    gt = load_ground_truth()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    start = time.time()
    page_filter = set(args.pages) if args.pages else None
    all_results = run_comparison(gt, args.patent, page_filter)
    elapsed = time.time() - start

    print(f"\nTotal wall time: {elapsed:.1f}s")
    print_scorecard(all_results)


if __name__ == "__main__":
    main()
