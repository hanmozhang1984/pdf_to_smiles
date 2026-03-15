#!/usr/bin/env python3
"""Evaluation harness for structure detection.

Runs the DoclingDetector on ground-truth pages, compares detected counts
to expected counts, saves annotated images, and prints a scorecard.

Usage:
    python eval/run_eval.py                    # run all patents
    python eval/run_eval.py --patent GLP       # run only GLP patent
    python eval/run_eval.py --patent Merck     # run only Merck patent
    python eval/run_eval.py --patent GLP --pages 99 104 112  # specific pages
"""

import argparse
import json
import os
import sys
import time

# Add project src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pypdfium2 as pdfium
from PIL import Image, ImageDraw, ImageFont

from pdf_to_smiles.core.docling_classifier import DoclingClassifier
from pdf_to_smiles.core.docling_detector import DoclingDetector

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

PDF_DIR = os.path.expanduser(
    "~/Downloads/Sample patents for testing"
)
DPI_SCALE = 200 / 72
GROUND_TRUTH_PATH = os.path.join(os.path.dirname(__file__), "ground_truth.json")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def load_ground_truth():
    with open(GROUND_TRUTH_PATH) as f:
        return json.load(f)


def run_patent(patent_cfg, page_filter=None):
    """Run evaluation for one patent. Returns list of result dicts."""
    patent_id = patent_cfg["id"]
    pdf_path = os.path.join(PDF_DIR, patent_cfg["filename"])

    if not os.path.exists(pdf_path):
        print(f"  SKIP: {pdf_path} not found")
        return []

    # Create output directory for this patent
    patent_out = os.path.join(OUTPUT_DIR, patent_id)
    os.makedirs(patent_out, exist_ok=True)

    # Initialize classifier + detector (fresh per patent)
    print(f"  Initializing classifier for {patent_id}...")
    classifier = DoclingClassifier()
    classifier.classify_pdf(pdf_path)
    detector = DoclingDetector(classifier)
    doc = pdfium.PdfDocument(pdf_path)

    pages = patent_cfg["pages"]
    if page_filter:
        pages = [p for p in pages if p["page"] in page_filter]

    results = []
    for page_cfg in pages:
        page_num = page_cfg["page"]
        expected = page_cfg["expected_count"]

        if page_num > len(doc):
            print(f"  Page {page_num} exceeds document length ({len(doc)}), skipping")
            continue

        # Render and detect
        page = doc[page_num - 1]
        bitmap = page.render(scale=DPI_SCALE)
        pil_image = bitmap.to_pil()

        detections = detector.detect_structures_with_boxes(pil_image, page_num)
        detected = len(detections)

        # Compute result
        delta = detected - expected
        if delta == 0:
            status = "PASS"
        elif delta > 0:
            status = f"OVER(+{delta})"
        else:
            status = f"MISS({delta})"

        result = {
            "patent": patent_id,
            "page": page_num,
            "expected": expected,
            "detected": detected,
            "delta": delta,
            "status": status,
            "examples": page_cfg.get("examples", ""),
            "notes": page_cfg.get("notes", ""),
        }
        results.append(result)

        # Save annotated image
        annotated = pil_image.copy()
        draw = ImageDraw.Draw(annotated)
        for i, (crop_img, box) in enumerate(detections):
            x1, y1, x2, y2 = box
            color = (0, 200, 0) if delta == 0 else (255, 0, 0)
            draw.rectangle(box, outline=color, width=3)
            label = f"[{i}] {x2-x1}x{y2-y1}"
            draw.text((x1 + 4, y1 + 4), label, fill=color)

        # Add status bar at top
        status_color = (0, 180, 0) if delta == 0 else (220, 0, 0)
        status_text = f"Page {page_num} | Expected: {expected} | Detected: {detected} | {status}"
        draw.rectangle([(0, 0), (pil_image.width, 28)], fill=(40, 40, 40))
        draw.text((8, 6), status_text, fill=status_color)

        annotated.save(os.path.join(patent_out, f"page_{page_num:03d}.png"))

    return results


def print_scorecard(all_results):
    """Print summary scorecard."""
    if not all_results:
        print("\nNo results to report.")
        return

    # Group by patent
    patents = {}
    for r in all_results:
        patents.setdefault(r["patent"], []).append(r)

    print("\n" + "=" * 80)
    print("EVALUATION SCORECARD")
    print("=" * 80)

    total_pass = 0
    total_pages = 0
    total_expected = 0
    total_detected = 0

    for patent_id, results in patents.items():
        print(f"\n--- {patent_id} ---")
        print(f"{'Page':>6} {'Expected':>8} {'Detected':>8} {'Delta':>6} {'Status':>10}  {'Examples'}")
        print("-" * 70)

        patent_pass = 0
        for r in results:
            marker = "  " if r["delta"] == 0 else ">>"
            print(
                f"{marker} {r['page']:4d} {r['expected']:8d} {r['detected']:8d} "
                f"{r['delta']:+5d}  {r['status']:>10}  {r['examples']}"
            )
            if r["delta"] == 0:
                patent_pass += 1
            total_expected += r["expected"]
            total_detected += r["detected"]

        total_pass += patent_pass
        total_pages += len(results)
        pct = patent_pass / len(results) * 100 if results else 0
        print(f"\n  {patent_id}: {patent_pass}/{len(results)} pages correct ({pct:.0f}%)")

    print("\n" + "=" * 80)
    pct = total_pass / total_pages * 100 if total_pages else 0
    print(f"OVERALL: {total_pass}/{total_pages} pages with exact count ({pct:.0f}%)")
    print(f"Total structures: expected={total_expected}, detected={total_detected}, "
          f"delta={total_detected - total_expected:+d}")
    print("=" * 80)

    # Save results as JSON
    results_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed results saved to {results_path}")
    print(f"Annotated images saved to {OUTPUT_DIR}/")


def main():
    parser = argparse.ArgumentParser(description="Structure detection evaluation harness")
    parser.add_argument("--patent", type=str, help="Run only this patent ID (e.g., GLP, Merck)")
    parser.add_argument("--pages", type=int, nargs="+", help="Run only these PDF page numbers")
    args = parser.parse_args()

    gt = load_ground_truth()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = []
    start = time.time()

    for patent_cfg in gt["patents"]:
        if args.patent and patent_cfg["id"] != args.patent:
            continue

        print(f"\n{'='*40}")
        print(f"Processing patent: {patent_cfg['id']}")
        print(f"{'='*40}")

        page_filter = set(args.pages) if args.pages else None
        results = run_patent(patent_cfg, page_filter)
        all_results.extend(results)

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")

    print_scorecard(all_results)


if __name__ == "__main__":
    main()
