#!/usr/bin/env python3
"""Pilot comparison: shrink-for-detection vs shrink-final-crops.

Approach A: Downscale page image before detection, map box coords back
            to original resolution, crop from full-res page.
Approach B: Detect and crop at full resolution, then resize oversized
            crops before SMILES prediction.
Baseline:   No shrinking — current pipeline as-is.

Runs on GLP/GLP1 problem pages + HER2/KAT6 for regression check.
Generates side-by-side crops and SMILES predictions.

Usage:
    python eval/pilot_shrink.py
    python eval/pilot_shrink.py --patent GLP --pages 100
"""

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pypdfium2 as pdfium
from PIL import Image

from pdf_to_smiles.core.docling_classifier import DoclingClassifier
from pdf_to_smiles.core.docling_detector import DoclingDetector

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

PDF_DIR = os.path.expanduser("~/Downloads/Sample patents for testing")
DPI_SCALE = 200 / 72
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output", "pilot_shrink")

# Detection downscale factor for Approach A
DETECTION_SCALE = 0.6  # 60% of original → structures become ~300-400px

# Max crop dimension for Approach B
MAX_CROP_DIM = 600  # resize crops larger than this

# Test pages: (patent_id, filename, [pages])
TEST_PAGES = [
    ("GLP", "US20240366598A1.pdf", [98, 100, 105, 112]),
    ("GLP1", "US10934279.pdf", [93, 95]),
    ("HER2", "US20230041385A1.pdf", [85, 86]),
    ("KAT6", "US11492346.pdf", [1, 2]),
]


def detect_baseline(detector, page_image, page_num):
    """Baseline: detect at full resolution."""
    return detector.detect_structures_with_boxes(page_image, page_num)


def detect_approach_a(classifier, page_image, page_num, scale=DETECTION_SCALE):
    """Approach A: detect on downscaled image, crop from original.

    Returns list of (crop_from_original, box_in_original) tuples.
    """
    orig_w, orig_h = page_image.size
    small_w = int(orig_w * scale)
    small_h = int(orig_h * scale)
    small_image = page_image.resize((small_w, small_h), Image.LANCZOS)

    # Fresh detector for the downscaled image (caches are size-dependent)
    detector_small = DoclingDetector(classifier)

    # Detect on small image
    detections_small = detector_small.detect_structures_with_boxes(
        small_image, page_num
    )

    # Map boxes back to original resolution and crop from original
    results = []
    inv_scale = 1.0 / scale
    for crop_small, (sx1, sy1, sx2, sy2) in detections_small:
        ox1 = max(0, int(sx1 * inv_scale))
        oy1 = max(0, int(sy1 * inv_scale))
        ox2 = min(orig_w, int(sx2 * inv_scale))
        oy2 = min(orig_h, int(sy2 * inv_scale))
        crop_orig = page_image.crop((ox1, oy1, ox2, oy2))
        results.append((crop_orig, (ox1, oy1, ox2, oy2)))

    return results


def shrink_crop(crop_image, max_dim=MAX_CROP_DIM):
    """Approach B helper: shrink a crop if it exceeds max_dim."""
    w, h = crop_image.size
    if max(w, h) <= max_dim:
        return crop_image  # already small enough
    ratio = max_dim / max(w, h)
    new_w = int(w * ratio)
    new_h = int(h * ratio)
    return crop_image.resize((new_w, new_h), Image.LANCZOS)


def detect_approach_b(detector, page_image, page_num):
    """Approach B: detect at full res, shrink oversized crops."""
    detections = detector.detect_structures_with_boxes(page_image, page_num)
    results = []
    for crop_img, box in detections:
        shrunk = shrink_crop(crop_img)
        results.append((shrunk, box))
    return results


def predict_smiles(crop_image, predictor):
    """Run SMILES prediction on a crop image.

    Uses predict_single (raw MolSight output) since RDKit may not be
    installed for validation. Falls back to predict() if predict_single
    is not available.
    """
    try:
        # Try raw prediction first (avoids RDKit validation issues)
        if hasattr(predictor, 'predict_single'):
            raw = predictor.predict_single(crop_image)
            if raw and raw != "NONE":
                return raw
            # Try cleaned variant
            if hasattr(predictor, '_preprocess'):
                cleaned = predictor._preprocess(crop_image)
                raw2 = predictor.predict_single(cleaned)
                if raw2 and raw2 != "NONE":
                    return raw2
            return raw
        return predictor.predict(crop_image)
    except Exception as e:
        return f"ERROR: {e}"


def run_comparison(args):
    """Run the full comparison across test pages."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Try to load MolSight predictor (preferred), fall back to MolScribe
    predictor = None
    predictor_name = "none"
    try:
        from pdf_to_smiles.core.molsight_predictor import MolSightPredictor
        predictor = MolSightPredictor()
        predictor_name = "molsight"
        print("Using MolSight predictor")
    except Exception:
        try:
            from pdf_to_smiles.core.lightweight_predictor import LightweightPredictor
            predictor = LightweightPredictor()
            predictor_name = "molscribe"
            print("Using MolScribe predictor")
        except Exception:
            print("WARNING: No SMILES predictor available, crops only")

    results = []

    for patent_id, filename, pages in TEST_PAGES:
        if args.patent and patent_id != args.patent:
            continue

        pdf_path = os.path.join(PDF_DIR, filename)
        if not os.path.exists(pdf_path):
            print(f"  SKIP: {pdf_path} not found")
            continue

        page_list = pages
        if args.pages:
            page_list = [p for p in pages if p in args.pages]
        if not page_list:
            continue

        print(f"\n{'='*50}")
        print(f"Patent: {patent_id}")
        print(f"{'='*50}")

        # Initialize classifier once per patent
        classifier = DoclingClassifier()
        classifier.classify_pdf(pdf_path)
        doc = pdfium.PdfDocument(pdf_path)

        patent_dir = os.path.join(OUTPUT_DIR, patent_id)
        os.makedirs(patent_dir, exist_ok=True)

        for page_num in page_list:
            if page_num > len(doc):
                continue

            page = doc[page_num - 1]
            bitmap = page.render(scale=DPI_SCALE)
            page_image = bitmap.to_pil()

            print(f"\n  Page {page_num} ({page_image.size[0]}x{page_image.size[1]}):")

            # --- Baseline ---
            detector_base = DoclingDetector(classifier)
            t0 = time.time()
            dets_base = detect_baseline(detector_base, page_image, page_num)
            t_base = time.time() - t0
            print(f"    Baseline:    {len(dets_base)} structures ({t_base:.1f}s)")

            # --- Approach A: shrink for detection ---
            t0 = time.time()
            dets_a = detect_approach_a(classifier, page_image, page_num)
            t_a = time.time() - t0
            print(f"    Approach A:  {len(dets_a)} structures ({t_a:.1f}s)")

            # --- Approach B: shrink final crops ---
            detector_b = DoclingDetector(classifier)
            t0 = time.time()
            dets_b = detect_approach_b(detector_b, page_image, page_num)
            t_b = time.time() - t0
            print(f"    Approach B:  {len(dets_b)} structures ({t_b:.1f}s)")

            # Save crops and run predictions
            approaches = [
                ("baseline", dets_base),
                ("approach_a", dets_a),
                ("approach_b", dets_b),
            ]

            for approach_name, detections in approaches:
                for i, (crop_img, box) in enumerate(detections):
                    # Save crop
                    crop_path = os.path.join(
                        patent_dir,
                        f"p{page_num:03d}_{approach_name}_s{i}.png",
                    )
                    crop_img.save(crop_path)

                    # Predict SMILES
                    smiles = None
                    if predictor:
                        smiles = predict_smiles(crop_img, predictor)

                    w, h = crop_img.size
                    result = {
                        "patent": patent_id,
                        "page": page_num,
                        "approach": approach_name,
                        "struct_idx": i,
                        "box": list(box),
                        "crop_w": w,
                        "crop_h": h,
                        "smiles": smiles,
                        "predictor": predictor_name,
                    }
                    results.append(result)

                    tag = f"{'✓' if smiles else '✗'}"
                    print(
                        f"      {approach_name}[{i}]: {w}x{h}  "
                        f"{tag} {(smiles or '')[:60]}"
                    )

    # Save results
    results_path = os.path.join(OUTPUT_DIR, "comparison.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save CSV summary
    csv_path = os.path.join(OUTPUT_DIR, "comparison.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "patent", "page", "approach", "struct_idx",
                "crop_w", "crop_h", "smiles", "predictor",
            ],
        )
        writer.writeheader()
        for r in results:
            row = {k: r[k] for k in writer.fieldnames}
            writer.writerow(row)

    print(f"\n{'='*60}")
    print(f"Results saved to {results_path}")
    print(f"CSV saved to {csv_path}")
    print(f"Crops saved to {OUTPUT_DIR}/")
    print(f"{'='*60}")

    # Print summary table
    print_summary(results)


def print_summary(results):
    """Print a compact comparison summary."""
    if not results:
        return

    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")

    # Group by patent/page
    groups = {}
    for r in results:
        key = (r["patent"], r["page"])
        groups.setdefault(key, {}).setdefault(r["approach"], []).append(r)

    print(
        f"\n{'Patent':>8} {'Page':>5} │ {'Approach':>12} {'Count':>6} "
        f"{'Avg W':>6} {'Avg H':>6} {'Valid':>6}"
    )
    print("─" * 70)

    for (patent, page), approaches in sorted(groups.items()):
        first = True
        for approach_name in ["baseline", "approach_a", "approach_b"]:
            if approach_name not in approaches:
                continue
            items = approaches[approach_name]
            count = len(items)
            avg_w = sum(r["crop_w"] for r in items) / max(count, 1)
            avg_h = sum(r["crop_h"] for r in items) / max(count, 1)
            valid = sum(1 for r in items if r["smiles"] and not str(r["smiles"]).startswith("ERROR"))

            label = f"{patent:>8} {page:>5}" if first else f"{'':>8} {'':>5}"
            print(
                f"{label} │ {approach_name:>12} {count:>6} "
                f"{avg_w:>6.0f} {avg_h:>6.0f} {valid:>5}/{count}"
            )
            first = False
        print("─" * 70)


def main():
    parser = argparse.ArgumentParser(description="Pilot: shrink comparison")
    parser.add_argument("--patent", type=str, help="Run only this patent")
    parser.add_argument("--pages", type=int, nargs="+", help="Run only these pages")
    parser.add_argument("--no-predict", action="store_true", help="Skip SMILES prediction")
    args = parser.parse_args()

    run_comparison(args)


if __name__ == "__main__":
    main()
