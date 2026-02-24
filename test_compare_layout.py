"""Compare current PageClassifier (pixel heuristic) vs DocLayout-YOLO on patent PDFs.

Runs both systems on a range of pages and compares:
1. Page classification accuracy (does the page have structures? tables?)
2. "Should process" binary metric (the actual decision we care about)
3. Detection granularity (what categories does each detect?)
4. Speed comparison
"""
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import pypdfium2 as pdfium
from PIL import Image
import numpy as np

# --- Load current PageClassifier ---
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from pdf_to_smiles.core.page_classifier import PageClassifier

# --- Load DocLayout-YOLO via our new classifier ---
from pdf_to_smiles.core.doclayout_classifier import DocLayoutClassifier

print("Loading DocLayout-YOLO model...")
doclayout = DocLayoutClassifier()
# Force model load now so timing is fair
doclayout._load_model()
print("Models loaded.\n")

classifier = PageClassifier()

# --- Test patents and pages ---
patent_dir = "/Users/hanmozhang/Downloads/Sample patents for testing"

# Ground truth: manually labeled pages (corrected labels)
# Categories: "structures", "table", "text_only", "mixed" (structures + text),
#             "markush" (generic structures with R-groups)
# should_process: True for structures/table/markush/mixed, False for text_only
ground_truth = {
    "WO_2026024861_A1.pdf": {
        1: "text_only",       # Title/abstract page - no structures
        5: "structures",      # Has Markush structure at top of page
        10: "structures",     # Multiple inline Markush structures
        20: "structures",     # Multiple inline Markush structures
        30: "text_only",      # Chemical names only, no structure drawings
        40: "text_only",      # Text-only (corrected: no structure drawings)
        50: "text_only",      # Text-only (corrected: no structure drawings)
        60: "table",          # Table (corrected: compound data table)
        70: "structures",     # Chemical structures
        80: "table",          # Structures at top + Table 5 with structures (corrected: table+structures)
        90: "structures",     # Chemical structures
        100: "structures",    # Intermediate structures
        110: "table",         # Table with structures (corrected: compound table)
        120: "table",         # Table with structures (corrected: compound table)
        130: "text_only",     # Pure synthesis text, no structure drawings (corrected)
        140: "structures",    # Example compound with structure drawing + synthesis text
        144: "table",         # Compound data table
        150: "table",         # Compound data table
        160: "structures",    # Structures with synthesis text
        170: "text_only",     # Synthesis text only, no structure drawings
        180: "structures",    # Grid of compound structures
        190: "structures",    # More structures
        196: "table",         # Table 17 — full page of compound structures in table rows (corrected)
    },
    "US11608344.pdf": {
        1: "structures",      # Cover page with small chemical structure in abstract (corrected)
        5: "structures",      # Figure page (bar chart = figure)
        10: "structures",     # Inline Markush structures in claims
        20: "structures",     # Inline Markush structures
        30: "markush",        # Markush structures with text
        40: "markush",        # Markush structures
        50: "markush",        # Markush fragments
        60: "structures",     # Chemical structures (TABLE A with structures)
        70: "structures",     # Chemical structures
        80: "structures",     # Chemical structures
        90: "structures",     # Compound table with structures (TABLE A)
        100: "structures",    # TABLE A continued with structures
    },
    "US10934279.pdf": {
        1: "text_only",       # Cover page
        5: "structures",      # Claims with inline Markush structures
        10: "text_only",      # Pure text chemical names, no drawings
        20: "structures",     # Inline structures in description
        30: "structures",     # Reaction scheme / synthesis
        40: "structures",     # Synthesis
        50: "structures",     # Synthesis
        60: "structures",     # Synthesis
        70: "structures",     # Structures with reaction schemes
    },
}


def classify_yolo_page(pil_image):
    """Classify using our DocLayoutClassifier."""
    result = doclayout.classify_page(pil_image)
    if result.has_tables:
        return "table", result.categories
    elif result.has_structures:
        return "structures", result.categories
    else:
        return "text_only", result.categories


def classify_heuristic(pil_image):
    """Classify using current PageClassifier heuristics."""
    has_struct = classifier._has_structure_graphics(pil_image)
    has_table = classifier._has_bio_table_indicators(pil_image)

    if has_table:
        return "table"
    elif has_struct:
        return "structures"
    else:
        return "text_only"


def should_process_gt(gt_label):
    """Whether ground truth says we should process this page."""
    return gt_label != "text_only"


def should_process_prediction(pred_label):
    """Whether prediction says we should process this page."""
    return pred_label != "text_only"


# --- Run comparison ---
results = []
out_dir = "/Users/hanmozhang/Downloads/test_image_cleanup/layout_comparison"
os.makedirs(out_dir, exist_ok=True)

for pdf_name, page_labels in ground_truth.items():
    pdf_path = os.path.join(patent_dir, pdf_name)
    if not os.path.exists(pdf_path):
        print(f"Skipping {pdf_name} (not found)")
        continue

    doc = pdfium.PdfDocument(pdf_path)
    total = len(doc)
    print(f"\n{'='*80}")
    print(f"{pdf_name} ({total} pages)")
    print(f"{'='*80}")
    print(f"{'Page':>6}  {'Ground Truth':>12}  {'Heuristic':>12}  {'YOLO':>12}  {'H?':>3}  {'Y?':>3}  {'Proc?':>5}  YOLO categories")
    print("-" * 110)

    for page_num, gt_label in sorted(page_labels.items()):
        if page_num > total:
            continue

        # Render page
        page = doc[page_num - 1]

        # Low-res for heuristic (72 DPI)
        bitmap_low = page.render(scale=72/72)
        pil_low = bitmap_low.to_pil()

        # Higher-res for YOLO (200 DPI)
        bitmap_hi = page.render(scale=200/72)
        pil_hi = bitmap_hi.to_pil()

        # --- Heuristic classification ---
        t0 = time.time()
        h_label = classify_heuristic(pil_low)
        h_time = time.time() - t0

        # --- YOLO classification ---
        t0 = time.time()
        y_label, y_cats = classify_yolo_page(pil_hi)
        y_time = time.time() - t0

        # Compare to ground truth (exact category match)
        # "markush" counts as "structures" for comparison
        gt_normalized = "structures" if gt_label in ("markush", "mixed") else gt_label

        h_correct = "Y" if h_label == gt_normalized else "N"
        y_correct = "Y" if y_label == gt_normalized else "N"

        # "Should process" binary metric
        gt_proc = should_process_gt(gt_label)
        y_proc = should_process_prediction(y_label)
        h_proc = should_process_prediction(h_label)
        proc_match = "Y" if y_proc == gt_proc else "N"

        cat_str = ", ".join(f"{k}:{v}" for k, v in sorted(y_cats.items()))

        print(f"{page_num:>6}  {gt_label:>12}  {h_label:>12}  {y_label:>12}  {h_correct:>3}  {y_correct:>3}  {proc_match:>5}  {cat_str}")

        results.append({
            "pdf": pdf_name,
            "page": page_num,
            "ground_truth": gt_label,
            "gt_normalized": gt_normalized,
            "heuristic": h_label,
            "yolo": y_label,
            "h_correct": h_correct == "Y",
            "y_correct": y_correct == "Y",
            "gt_should_process": gt_proc,
            "h_should_process": h_proc,
            "y_should_process": y_proc,
            "h_time": h_time,
            "y_time": y_time,
            "yolo_categories": y_cats,
        })

    doc.close()

# --- Summary statistics ---
print(f"\n\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")

total = len(results)
h_correct = sum(1 for r in results if r["h_correct"])
y_correct = sum(1 for r in results if r["y_correct"])

print(f"Total pages tested: {total}")
print(f"\n--- Exact Category Accuracy ---")
print(f"Heuristic accuracy: {h_correct}/{total} = {100*h_correct/total:.1f}%")
print(f"YOLO accuracy:      {y_correct}/{total} = {100*y_correct/total:.1f}%")

# "Should process" binary metric — the metric that actually matters
print(f"\n--- Should Process (Binary) Accuracy ---")
h_proc_correct = sum(1 for r in results if r["h_should_process"] == r["gt_should_process"])
y_proc_correct = sum(1 for r in results if r["y_should_process"] == r["gt_should_process"])
print(f"Heuristic: {h_proc_correct}/{total} = {100*h_proc_correct/total:.1f}%")
print(f"YOLO:      {y_proc_correct}/{total} = {100*y_proc_correct/total:.1f}%")

# Precision/recall for "should process"
gt_positive = [r for r in results if r["gt_should_process"]]
gt_negative = [r for r in results if not r["gt_should_process"]]

for name, pred_key in [("Heuristic", "h_should_process"), ("YOLO", "y_should_process")]:
    tp = sum(1 for r in gt_positive if r[pred_key])
    fn = sum(1 for r in gt_positive if not r[pred_key])
    fp = sum(1 for r in gt_negative if r[pred_key])
    tn = sum(1 for r in gt_negative if not r[pred_key])
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    print(f"\n  {name}:")
    print(f"    TP={tp} FN={fn} FP={fp} TN={tn}")
    print(f"    Precision: {100*precision:.1f}%  Recall: {100*recall:.1f}%  F1: {100*f1:.1f}%")

# Per-category breakdown
print(f"\n--- Per-Category Breakdown ---")
for cat in ["text_only", "structures", "table"]:
    cat_results = [r for r in results if r["gt_normalized"] == cat]
    if not cat_results:
        continue
    h_cat = sum(1 for r in cat_results if r["h_correct"])
    y_cat = sum(1 for r in cat_results if r["y_correct"])
    n = len(cat_results)
    print(f"\n  {cat} pages ({n}):")
    print(f"    Heuristic: {h_cat}/{n} = {100*h_cat/n:.1f}%")
    print(f"    YOLO:      {y_cat}/{n} = {100*y_cat/n:.1f}%")

# Speed comparison
h_avg = np.mean([r["h_time"] for r in results]) * 1000
y_avg = np.mean([r["y_time"] for r in results]) * 1000
print(f"\nAverage time per page:")
print(f"  Heuristic: {h_avg:.0f} ms")
print(f"  YOLO:      {y_avg:.0f} ms")

# Disagreement analysis
disagreements = [r for r in results if r["h_correct"] != r["y_correct"]]
if disagreements:
    print(f"\nDisagreements ({len(disagreements)} pages):")
    for r in disagreements:
        winner = "YOLO" if r["y_correct"] else "Heuristic"
        print(f"  {r['pdf']} p{r['page']}: GT={r['ground_truth']}, "
              f"H={r['heuristic']}, Y={r['yolo']} -> {winner} wins")

# Errors by each system
print(f"\nHeuristic errors ({total - h_correct}):")
for r in results:
    if not r["h_correct"]:
        print(f"  {r['pdf']} p{r['page']}: GT={r['gt_normalized']}, predicted={r['heuristic']}")

print(f"\nYOLO errors ({total - y_correct}):")
for r in results:
    if not r["y_correct"]:
        print(f"  {r['pdf']} p{r['page']}: GT={r['gt_normalized']}, predicted={r['yolo']}")

# "Should process" errors
print(f"\nYOLO 'should process' errors ({total - y_proc_correct}):")
for r in results:
    if r["y_should_process"] != r["gt_should_process"]:
        direction = "FP (processed text-only)" if r["y_should_process"] else "FN (missed structure)"
        print(f"  {r['pdf']} p{r['page']}: GT={r['ground_truth']}, YOLO={r['yolo']} — {direction}")

print("\nDone.")
