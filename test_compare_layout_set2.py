"""Test set 2: Head-to-head comparison of three classification protocols.

1. Heuristic (PageClassifier) — pixel-level dark pixel spread analysis
2. Raw DocLayout-YOLO — original YOLO classification logic (figure=structures, table=table, else=text_only)
3. Modified DocLayout-YOLO — our optimized classification (include tables + formula categories)

Tests on curated single-page PDFs from Doclayout_YOLO_test_set_2.
"""
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import pypdfium2 as pdfium
from PIL import Image
import numpy as np

# --- Load classifiers ---
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from pdf_to_smiles.core.page_classifier import PageClassifier
from pdf_to_smiles.core.doclayout_classifier import DocLayoutClassifier

print("Loading models...")
doclayout = DocLayoutClassifier()
doclayout._load_model()
heuristic = PageClassifier()
print("Models loaded.\n")

# --- Test set 2 ---
test_dir = "/Users/hanmozhang/Downloads/Doclayout_YOLO_test_set_2"

# Ground truth for test set 2 (verified by visual inspection)
# Format: {filename: {page_num(1-indexed): label}}
ground_truth = {
    "US11492346_page38.pdf": {
        1: "structures",     # Structures + table (TABLE 10 with compound structures + reaction schemes)
    },
    "US11608344_page272.pdf": {
        1: "structures",     # TABLE 7 with compound structures + synthesis example
    },
    "US20230041385A1_page226.pdf": {
        1: "structures",     # Claims page with Markush structure + many fragment structures
    },
    "US20240366598A1_page86.pdf": {
        1: "structures",     # Reaction scheme page (synthesis steps C55-C57 with structures)
    },
    "WO_2026024861_A1_page122_124.pdf": {
        1: "table",          # Table 11 with compound structures
        2: "structures",     # Structures with reaction schemes + synthesis text
        3: "text_only",      # Pure synthesis text, no structure drawings
    },
}

# --- Raw YOLO classification (original logic from first test script) ---
# This uses only "figure" as structure indicator, no formula categories
RAW_STRUCTURE_CATEGORIES = {"figure"}
RAW_TABLE_CATEGORIES = {"table"}
RAW_FORMULA_CATEGORIES = {"isolate_formula", "formula_caption"}


def get_yolo_categories(pil_image):
    """Run YOLO and return raw detection categories."""
    result = doclayout.classify_page(pil_image)
    return result.categories


def classify_raw_yolo(categories):
    """Original YOLO classification logic (from first test script).

    - table detected -> "table"
    - figure OR formula detected -> "structures"
    - else -> "text_only"
    """
    has_figures = any(c in categories for c in RAW_STRUCTURE_CATEGORIES)
    has_tables = any(c in categories for c in RAW_TABLE_CATEGORIES)
    has_formulas = any(c in categories for c in RAW_FORMULA_CATEGORIES)

    if has_tables:
        return "table"
    elif has_figures or has_formulas:
        return "structures"
    else:
        return "text_only"


def classify_modified_yolo(categories):
    """Modified YOLO classification (our optimized logic).

    Key difference: both tables AND structures -> should_process.
    Only pure text pages are skipped.
    - figure, isolate_formula, formula_caption -> "structures"
    - table -> "table"
    - only plain text/title/abandon -> "text_only"
    """
    from pdf_to_smiles.core.doclayout_classifier import (
        _STRUCTURE_CATEGORIES, _TABLE_CATEGORIES,
    )

    has_structures = any(c in categories for c in _STRUCTURE_CATEGORIES)
    has_tables = any(c in categories for c in _TABLE_CATEGORIES)

    if has_tables:
        return "table"
    elif has_structures:
        return "structures"
    else:
        return "text_only"


def classify_heuristic(pil_image):
    """Classify using pixel heuristic."""
    has_struct = heuristic._has_structure_graphics(pil_image)
    has_table = heuristic._has_bio_table_indicators(pil_image)

    if has_table:
        return "table"
    elif has_struct:
        return "structures"
    else:
        return "text_only"


def should_process(label):
    """Binary: should we process this page?"""
    return label != "text_only"


# --- Run comparison ---
results = []

for pdf_name, page_labels in ground_truth.items():
    pdf_path = os.path.join(test_dir, pdf_name)
    if not os.path.exists(pdf_path):
        print(f"Skipping {pdf_name} (not found)")
        continue

    doc = pdfium.PdfDocument(pdf_path)
    total = len(doc)
    print(f"\n{'='*100}")
    print(f"{pdf_name} ({total} pages)")
    print(f"{'='*100}")
    print(f"{'Page':>4}  {'GT':>10}  {'Heuristic':>10}  {'RawYOLO':>10}  {'ModYOLO':>10}  "
          f"{'H?':>3}  {'R?':>3}  {'M?':>3}  YOLO categories")
    print("-" * 120)

    for page_num, gt_label in sorted(page_labels.items()):
        if page_num > total:
            continue

        page = doc[page_num - 1]

        # Low-res for heuristic (72 DPI)
        bitmap_low = page.render(scale=72/72)
        pil_low = bitmap_low.to_pil()

        # Higher-res for YOLO (200 DPI)
        bitmap_hi = page.render(scale=200/72)
        pil_hi = bitmap_hi.to_pil()

        # --- Heuristic ---
        t0 = time.time()
        h_label = classify_heuristic(pil_low)
        h_time = time.time() - t0

        # --- YOLO (get categories once, classify two ways) ---
        t0 = time.time()
        y_cats = get_yolo_categories(pil_hi)
        y_time = time.time() - t0

        raw_label = classify_raw_yolo(y_cats)
        mod_label = classify_modified_yolo(y_cats)

        # Normalize GT for comparison
        gt_normalized = "structures" if gt_label in ("markush", "mixed") else gt_label

        h_match = "Y" if h_label == gt_normalized else "N"
        r_match = "Y" if raw_label == gt_normalized else "N"
        m_match = "Y" if mod_label == gt_normalized else "N"

        cat_str = ", ".join(f"{k}:{v}" for k, v in sorted(y_cats.items()))

        print(f"{page_num:>4}  {gt_label:>10}  {h_label:>10}  {raw_label:>10}  {mod_label:>10}  "
              f"{h_match:>3}  {r_match:>3}  {m_match:>3}  {cat_str}")

        results.append({
            "pdf": pdf_name,
            "page": page_num,
            "ground_truth": gt_label,
            "gt_normalized": gt_normalized,
            "heuristic": h_label,
            "raw_yolo": raw_label,
            "mod_yolo": mod_label,
            "h_correct": h_label == gt_normalized,
            "r_correct": raw_label == gt_normalized,
            "m_correct": mod_label == gt_normalized,
            "gt_should_process": should_process(gt_label),
            "h_should_process": should_process(h_label),
            "r_should_process": should_process(raw_label),
            "m_should_process": should_process(mod_label),
            "h_time": h_time,
            "y_time": y_time,
            "yolo_categories": y_cats,
        })

    doc.close()

# --- Summary ---
print(f"\n\n{'='*80}")
print("TEST SET 2 — SUMMARY")
print(f"{'='*80}")

total = len(results)
h_correct = sum(1 for r in results if r["h_correct"])
r_correct = sum(1 for r in results if r["r_correct"])
m_correct = sum(1 for r in results if r["m_correct"])

print(f"Total pages tested: {total}")

print(f"\n--- Exact Category Accuracy ---")
print(f"  Heuristic:      {h_correct}/{total} = {100*h_correct/total:.1f}%")
print(f"  Raw YOLO:       {r_correct}/{total} = {100*r_correct/total:.1f}%")
print(f"  Modified YOLO:  {m_correct}/{total} = {100*m_correct/total:.1f}%")

print(f"\n--- Should Process (Binary) Accuracy ---")
h_proc = sum(1 for r in results if r["h_should_process"] == r["gt_should_process"])
r_proc = sum(1 for r in results if r["r_should_process"] == r["gt_should_process"])
m_proc = sum(1 for r in results if r["m_should_process"] == r["gt_should_process"])
print(f"  Heuristic:      {h_proc}/{total} = {100*h_proc/total:.1f}%")
print(f"  Raw YOLO:       {r_proc}/{total} = {100*r_proc/total:.1f}%")
print(f"  Modified YOLO:  {m_proc}/{total} = {100*m_proc/total:.1f}%")

# Precision/recall for "should process"
gt_positive = [r for r in results if r["gt_should_process"]]
gt_negative = [r for r in results if not r["gt_should_process"]]

for name, pred_key in [("Heuristic", "h_should_process"),
                       ("Raw YOLO", "r_should_process"),
                       ("Modified YOLO", "m_should_process")]:
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
    r_cat = sum(1 for r in cat_results if r["r_correct"])
    m_cat = sum(1 for r in cat_results if r["m_correct"])
    n = len(cat_results)
    print(f"\n  {cat} pages ({n}):")
    print(f"    Heuristic:      {h_cat}/{n} = {100*h_cat/n:.1f}%")
    print(f"    Raw YOLO:       {r_cat}/{n} = {100*r_cat/n:.1f}%")
    print(f"    Modified YOLO:  {m_cat}/{n} = {100*m_cat/n:.1f}%")

# Speed
h_avg = np.mean([r["h_time"] for r in results]) * 1000
y_avg = np.mean([r["y_time"] for r in results]) * 1000
print(f"\nAverage time per page:")
print(f"  Heuristic: {h_avg:.0f} ms")
print(f"  YOLO:      {y_avg:.0f} ms")

# Detailed errors
for name, correct_key, pred_key in [("Heuristic", "h_correct", "heuristic"),
                                      ("Raw YOLO", "r_correct", "raw_yolo"),
                                      ("Modified YOLO", "m_correct", "mod_yolo")]:
    errors = [r for r in results if not r[correct_key]]
    print(f"\n{name} errors ({len(errors)}):")
    for r in errors:
        print(f"  {r['pdf']} p{r['page']}: GT={r['gt_normalized']}, predicted={r[pred_key]}")

# "Should process" errors for each
for name, proc_key in [("Heuristic", "h_should_process"),
                       ("Raw YOLO", "r_should_process"),
                       ("Modified YOLO", "m_should_process")]:
    errors = [r for r in results if r[proc_key] != r["gt_should_process"]]
    print(f"\n{name} 'should process' errors ({len(errors)}):")
    for r in errors:
        direction = "FP (processed text-only)" if r[proc_key] else "FN (missed structure)"
        print(f"  {r['pdf']} p{r['page']}: GT={r['ground_truth']}, predicted={r[name.lower().replace(' ', '_') if 'YOLO' not in name else ('raw_yolo' if 'Raw' in name else 'mod_yolo')]} — {direction}")

print("\nDone.")
