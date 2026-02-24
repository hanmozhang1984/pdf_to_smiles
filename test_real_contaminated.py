"""Test text masking + image cleaning on real contaminated structure crops.

Runs three passes on each image:
  1. Text masking only (mask_text_regions)
  2. Image cleaning only (clean_structure_image, mask_text=False)
  3. Full pipeline (clean_structure_image, mask_text=True)

Saves before/after images and reports OCR detections + ink removal stats.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Ensure project is importable
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pdf_to_smiles.core.text_masker import mask_text_regions, _get_ocr, _is_atom_label
from pdf_to_smiles.core.image_cleaner import clean_structure_image


INPUT_DIR = Path.home() / "Downloads" / "Test_image_cleanup" / "Textorgraph_contaminating_structures"
OUTPUT_DIR = INPUT_DIR / "results"


def ink_count(img: Image.Image) -> int:
    arr = np.array(img.convert("L"))
    _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return int(cv2.countNonZero(binary))


def run_ocr_report(img: Image.Image, label: str) -> list[dict]:
    """Run PaddleOCR on an image and print what it finds."""
    arr = np.array(img if img.mode == "RGB" else img.convert("RGB"))
    img_h, img_w = arr.shape[:2]
    ocr = _get_ocr()
    results = list(ocr.predict(arr))

    detections = []
    if results:
        result = results[0]
        rec_texts = result.get("rec_texts", [])
        rec_scores = result.get("rec_scores", [])
        rec_polys = result.get("dt_polys", [])

        for text, score, poly in zip(rec_texts, rec_scores, rec_polys):
            poly_arr = np.array(poly)
            xs = poly_arr[:, 0]
            box_w = int(xs.max() - xs.min())
            is_atom = _is_atom_label(text, box_w, img_w)
            action = "KEEP (atom label)" if is_atom else "MASK"
            detections.append({
                "text": text,
                "confidence": score,
                "box_width": box_w,
                "action": action,
            })

    if detections:
        print(f"  [{label}] OCR detections:")
        for d in detections:
            print(f"    {d['action']:20s}  \"{d['text']}\"  (conf={d['confidence']:.2f}, box_w={d['box_width']}px)")
    else:
        print(f"  [{label}] No text detected by OCR")

    return detections


def process_image(image_path: Path):
    """Process a single contaminated image through all three passes."""
    name = image_path.stem
    print(f"\n{'─' * 70}")
    print(f"  {image_path.name}")
    print(f"{'─' * 70}")

    original = Image.open(image_path).convert("RGB")
    w, h = original.size
    original_ink = ink_count(original)
    print(f"  Size: {w}x{h}   Ink pixels: {original_ink}")

    # --- Step 1: OCR report on original ---
    detections = run_ocr_report(original, "ORIGINAL")

    text_to_mask = [d for d in detections if d["action"] == "MASK"]
    text_to_keep = [d for d in detections if d["action"] != "MASK"]
    print(f"  Summary: {len(text_to_mask)} regions to MASK, {len(text_to_keep)} atom labels to KEEP")

    # --- Step 2: Text masking only ---
    text_masked = mask_text_regions(original)
    tm_ink = ink_count(text_masked)
    tm_removed = 1.0 - tm_ink / max(original_ink, 1)
    print(f"\n  [TEXT MASK ONLY]  ink: {original_ink} → {tm_ink}  (removed {tm_removed:.1%})")

    # --- Step 3: Image cleaning only (no text mask) ---
    cleaned_no_text = clean_structure_image(original, mask_text=False)
    cn_ink = ink_count(cleaned_no_text)
    cn_removed = 1.0 - cn_ink / max(original_ink, 1)
    print(f"  [CLEAN ONLY]      ink: {original_ink} → {cn_ink}  (removed {cn_removed:.1%})")

    # --- Step 4: Full pipeline (text mask + clean) ---
    full_cleaned = clean_structure_image(original, mask_text=True)
    fc_ink = ink_count(full_cleaned)
    fc_removed = 1.0 - fc_ink / max(original_ink, 1)
    print(f"  [FULL PIPELINE]   ink: {original_ink} → {fc_ink}  (removed {fc_removed:.1%})")

    # --- Save results ---
    original.save(OUTPUT_DIR / f"{name}_0_original.png")
    text_masked.save(OUTPUT_DIR / f"{name}_1_text_masked.png")
    cleaned_no_text.save(OUTPUT_DIR / f"{name}_2_cleaned_only.png")
    full_cleaned.save(OUTPUT_DIR / f"{name}_3_full_pipeline.png")

    return {
        "name": name,
        "ocr_mask_count": len(text_to_mask),
        "ocr_keep_count": len(text_to_keep),
        "ink_original": original_ink,
        "ink_text_masked": tm_ink,
        "ink_cleaned_only": cn_ink,
        "ink_full_pipeline": fc_ink,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    images = sorted(INPUT_DIR.glob("*.png"))
    if not images:
        print(f"No PNG images found in {INPUT_DIR}")
        sys.exit(1)

    print("=" * 70)
    print(f"  Testing text masking on {len(images)} contaminated structure crops")
    print(f"  Output → {OUTPUT_DIR}")
    print("=" * 70)

    all_results = []
    for img_path in images:
        result = process_image(img_path)
        all_results.append(result)

    # --- Summary table ---
    print(f"\n\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Image':<45s} {'OCR→Mask':>8s} {'TextMask':>10s} {'CleanOnly':>10s} {'Full':>10s}")
    print(f"  {'':─<45s} {'':─>8s} {'':─>10s} {'':─>10s} {'':─>10s}")
    for r in all_results:
        tm_pct = f"{(1 - r['ink_text_masked'] / max(r['ink_original'], 1)) * 100:.1f}%"
        cn_pct = f"{(1 - r['ink_cleaned_only'] / max(r['ink_original'], 1)) * 100:.1f}%"
        fc_pct = f"{(1 - r['ink_full_pipeline'] / max(r['ink_original'], 1)) * 100:.1f}%"
        short_name = r["name"][:43]
        print(f"  {short_name:<45s} {r['ocr_mask_count']:>8d} {tm_pct:>10s} {cn_pct:>10s} {fc_pct:>10s}")

    print(f"\n  Results saved to: {OUTPUT_DIR}")
    print(f"  Compare *_0_original.png vs *_3_full_pipeline.png for each image.")


if __name__ == "__main__":
    main()
