"""Evaluate Docling (Apache 2.0) vs DocLayout-YOLO (AGPL) on patent PDFs.

Side-by-side comparison to determine if Docling can replace YOLO for page
classification, removing the AGPL licensing burden.

Docling uses DocLayNet labels (picture, formula, table, etc.) which we map
to our existing PageClassification categories.
"""
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import pypdfium2 as pdfium
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from pdf_to_smiles.core.doclayout_classifier import PageClassification

try:
    from pdf_to_smiles.core.doclayout_classifier import DocLayoutClassifier
    # Check if YOLO model can actually load
    import importlib
    importlib.import_module("doclayout_yolo")
    YOLO_AVAILABLE = True
except (ImportError, Exception):
    YOLO_AVAILABLE = False
    print("NOTE: DocLayout-YOLO not available — running Docling-only evaluation.")

# ── Docling category mapping ────────────────────────────────────────────────
# DocLayNet labels → our classification groups
# Chemical structures in patents appear as "picture" or "formula" in DocLayNet.
_DOCLING_STRUCTURE_LABELS = {"picture", "formula", "chart"}
_DOCLING_TABLE_LABELS = {"table"}
_DOCLING_TEXT_LABELS = {
    "caption", "text", "title", "section_header", "list_item",
    "footnote", "page_footer", "page_header", "code", "paragraph",
    "reference", "document_index",
}

CONFIDENCE_THRESHOLD = 0.25  # Same as YOLO


# ── Docling wrapper ─────────────────────────────────────────────────────────
class DoclingPageClassifier:
    """Evaluate Docling for page classification (processes entire PDFs)."""

    def __init__(self):
        self._converter = None

    def _init_converter(self):
        """Lazy-init the Docling converter with layout-only settings."""
        if self._converter is not None:
            return

        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions

        pipeline_options = PdfPipelineOptions(
            do_ocr=False,              # Skip OCR — we only need layout
            do_table_structure=False,   # Skip table parsing — we only need detection
            do_formula_enrichment=False,
            do_code_enrichment=False,
            do_picture_classification=False,
            do_picture_description=False,
        )

        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            }
        )

    def classify_pdf(self, pdf_path: str) -> dict:
        """Classify all pages in a PDF.

        Returns:
            Dict mapping 1-indexed page number to PageClassification.
        """
        self._init_converter()

        result = self._converter.convert(pdf_path)

        classifications = {}
        for page in result.pages:
            page_no = page.page_no  # 1-indexed
            layout = page.predictions.layout

            if layout is None:
                classifications[page_no] = PageClassification()
                continue

            categories = {}
            confidence_scores = {}

            for cluster in layout.clusters:
                label = cluster.label.value  # e.g. "picture", "table"
                conf = cluster.confidence

                if conf < CONFIDENCE_THRESHOLD:
                    continue

                categories[label] = categories.get(label, 0) + 1
                confidence_scores[label] = max(
                    confidence_scores.get(label, 0.0), conf
                )

            has_structures = any(c in categories for c in _DOCLING_STRUCTURE_LABELS)
            has_tables = any(c in categories for c in _DOCLING_TABLE_LABELS)

            classifications[page_no] = PageClassification(
                has_structures=has_structures,
                has_tables=has_tables,
                is_text_only=not has_structures and not has_tables,
                categories=categories,
                confidence_scores=confidence_scores,
            )

        return classifications


# ── Ground truth (same as test_compare_layout.py) ───────────────────────────
patent_dir = "/Users/hanmozhang/Downloads/Sample patents for testing"

ground_truth = {
    "WO_2026024861_A1.pdf": {
        1: "text_only",
        5: "structures",
        10: "structures",
        20: "structures",
        30: "text_only",
        40: "text_only",
        50: "text_only",
        60: "table",
        70: "structures",
        80: "table",
        90: "structures",
        100: "structures",
        110: "table",
        120: "table",
        130: "text_only",
        140: "structures",
        144: "table",
        150: "table",
        160: "structures",
        170: "text_only",
        180: "structures",
        190: "structures",
        196: "table",
    },
    "US11608344.pdf": {
        1: "structures",
        5: "structures",
        10: "structures",
        20: "structures",
        30: "markush",
        40: "markush",
        50: "markush",
        60: "structures",
        70: "structures",
        80: "structures",
        90: "structures",
        100: "structures",
    },
    "US10934279.pdf": {
        1: "text_only",
        5: "structures",
        10: "text_only",
        20: "structures",
        30: "structures",
        40: "structures",
        50: "structures",
        60: "structures",
        70: "structures",
    },
}


def classify_docling_label(classification):
    """Convert PageClassification to category label."""
    if classification.has_tables:
        return "table"
    elif classification.has_structures:
        return "structures"
    else:
        return "text_only"


def classify_yolo_page(doclayout, pil_image):
    """Classify using DocLayoutClassifier."""
    result = doclayout.classify_page(pil_image)
    if result.has_tables:
        return "table", result.categories
    elif result.has_structures:
        return "structures", result.categories
    else:
        return "text_only", result.categories


def should_process(label):
    """Whether a label means the page should be processed."""
    return label != "text_only"


# ── Main evaluation ─────────────────────────────────────────────────────────
def main():
    # Load YOLO (optional)
    doclayout = None
    if YOLO_AVAILABLE:
        print("Loading DocLayout-YOLO model...")
        doclayout = DocLayoutClassifier()
        doclayout._load_model()
        print("YOLO loaded.\n")
    else:
        print("Skipping YOLO (not installed).\n")

    # Load Docling
    print("Initializing Docling converter...")
    docling_clf = DoclingPageClassifier()
    docling_clf._init_converter()
    print("Docling ready.\n")

    results = []
    out_dir = "/Users/hanmozhang/Downloads/test_image_cleanup/docling_eval"
    os.makedirs(out_dir, exist_ok=True)

    for pdf_name, page_labels in ground_truth.items():
        pdf_path = os.path.join(patent_dir, pdf_name)
        if not os.path.exists(pdf_path):
            print(f"Skipping {pdf_name} (not found)")
            continue

        # ── Docling: process entire PDF at once ──
        print(f"\nProcessing {pdf_name} with Docling...")
        t0 = time.time()
        docling_results = docling_clf.classify_pdf(pdf_path)
        docling_total_time = time.time() - t0

        doc = pdfium.PdfDocument(pdf_path)
        total = len(doc)
        docling_per_page_ms = (docling_total_time / total) * 1000

        print(f"  Docling: {docling_total_time:.1f}s total, {docling_per_page_ms:.0f}ms/page ({total} pages)")

        print(f"\n{'='*100}")
        print(f"{pdf_name} ({total} pages)")
        print(f"{'='*100}")
        if doclayout:
            print(f"{'Page':>6}  {'GT':>12}  {'YOLO':>12}  {'Docling':>12}  {'Y?':>3}  {'D?':>3}  {'Yproc':>5}  {'Dproc':>5}  YOLO cats | Docling cats")
        else:
            print(f"{'Page':>6}  {'GT':>12}  {'Docling':>12}  {'D?':>3}  {'Dproc':>5}  Docling cats")
        print("-" * 130)

        for page_num, gt_label in sorted(page_labels.items()):
            if page_num > total:
                continue

            # ── YOLO: needs rendered image (optional) ──
            y_label, y_cats, y_time = "n/a", {}, 0.0
            if doclayout:
                page = doc[page_num - 1]
                bitmap = page.render(scale=200/72)
                pil_hi = bitmap.to_pil()

                t0 = time.time()
                y_label, y_cats = classify_yolo_page(doclayout, pil_hi)
                y_time = time.time() - t0

            # ── Docling: already have results ──
            d_clf = docling_results.get(page_num, PageClassification())
            d_label = classify_docling_label(d_clf)
            d_cats = d_clf.categories

            # Normalize GT
            gt_normalized = "structures" if gt_label in ("markush", "mixed") else gt_label

            y_correct = "Y" if y_label == gt_normalized else "N"
            d_correct = "Y" if d_label == gt_normalized else "N"

            gt_proc = should_process(gt_label)
            y_proc = should_process(y_label)
            d_proc = should_process(d_label)
            y_proc_match = "Y" if y_proc == gt_proc else "N"
            d_proc_match = "Y" if d_proc == gt_proc else "N"

            y_cat_str = ", ".join(f"{k}:{v}" for k, v in sorted(y_cats.items()))
            d_cat_str = ", ".join(f"{k}:{v}" for k, v in sorted(d_cats.items()))

            if doclayout:
                print(f"{page_num:>6}  {gt_label:>12}  {y_label:>12}  {d_label:>12}  "
                      f"{y_correct:>3}  {d_correct:>3}  {y_proc_match:>5}  {d_proc_match:>5}  "
                      f"{y_cat_str} | {d_cat_str}")
            else:
                print(f"{page_num:>6}  {gt_label:>12}  {d_label:>12}  "
                      f"{d_correct:>3}  {d_proc_match:>5}  {d_cat_str}")

            results.append({
                "pdf": pdf_name,
                "page": page_num,
                "ground_truth": gt_label,
                "gt_normalized": gt_normalized,
                "yolo": y_label,
                "docling": d_label,
                "y_correct": y_correct == "Y" if doclayout else None,
                "d_correct": d_correct == "Y",
                "gt_should_process": gt_proc,
                "y_should_process": y_proc if doclayout else None,
                "d_should_process": d_proc,
                "y_time": y_time,
                "d_time_per_page": docling_per_page_ms / 1000,
                "yolo_categories": y_cats,
                "docling_categories": d_cats,
            })

        doc.close()

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    total = len(results)
    d_correct = sum(1 for r in results if r["d_correct"])

    print(f"Total pages tested: {total}")
    print(f"\n--- Exact Category Accuracy ---")
    if doclayout:
        y_correct = sum(1 for r in results if r["y_correct"])
        print(f"YOLO accuracy:    {y_correct}/{total} = {100*y_correct/total:.1f}%")
    print(f"Docling accuracy: {d_correct}/{total} = {100*d_correct/total:.1f}%")

    # Binary "should process" metric
    print(f"\n--- Should Process (Binary) Accuracy ---")
    d_proc_correct = sum(1 for r in results if r["d_should_process"] == r["gt_should_process"])
    if doclayout:
        y_proc_correct = sum(1 for r in results if r["y_should_process"] == r["gt_should_process"])
        print(f"YOLO:    {y_proc_correct}/{total} = {100*y_proc_correct/total:.1f}%")
    print(f"Docling: {d_proc_correct}/{total} = {100*d_proc_correct/total:.1f}%")

    # Precision/recall for "should process"
    gt_positive = [r for r in results if r["gt_should_process"]]
    gt_negative = [r for r in results if not r["gt_should_process"]]

    classifiers_to_eval = [("Docling", "d_should_process")]
    if doclayout:
        classifiers_to_eval.insert(0, ("YOLO", "y_should_process"))

    for name, pred_key in classifiers_to_eval:
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
        d_cat = sum(1 for r in cat_results if r["d_correct"])
        n = len(cat_results)
        print(f"\n  {cat} pages ({n}):")
        if doclayout:
            y_cat = sum(1 for r in cat_results if r["y_correct"])
            print(f"    YOLO:    {y_cat}/{n} = {100*y_cat/n:.1f}%")
        print(f"    Docling: {d_cat}/{n} = {100*d_cat/n:.1f}%")

    # Speed comparison
    d_avg = np.mean([r["d_time_per_page"] for r in results]) * 1000
    print(f"\nAverage time per page:")
    if doclayout:
        y_avg = np.mean([r["y_time"] for r in results]) * 1000
        print(f"  YOLO:    {y_avg:.0f} ms")
    print(f"  Docling: {d_avg:.0f} ms (amortized over full PDF)")

    # Disagreement analysis: YOLO vs Docling
    if doclayout:
        disagreements = [r for r in results if r["y_correct"] != r["d_correct"]]
        if disagreements:
            print(f"\n--- Disagreements ({len(disagreements)} pages) ---")
            for r in disagreements:
                winner = "Docling" if r["d_correct"] else "YOLO"
                print(f"  {r['pdf']} p{r['page']}: GT={r['ground_truth']}, "
                      f"YOLO={r['yolo']}, Docling={r['docling']} -> {winner} wins")

    # Docling-specific errors
    print(f"\nDocling errors ({total - d_correct}):")
    for r in results:
        if not r["d_correct"]:
            d_cat_str = ", ".join(f"{k}:{v}" for k, v in sorted(r["docling_categories"].items()))
            print(f"  {r['pdf']} p{r['page']}: GT={r['gt_normalized']}, "
                  f"predicted={r['docling']}, cats=[{d_cat_str}]")

    # "Should process" errors
    print(f"\nDocling 'should process' errors ({total - d_proc_correct}):")
    for r in results:
        if r["d_should_process"] != r["gt_should_process"]:
            direction = "FP (processed text-only)" if r["d_should_process"] else "FN (missed structure/table)"
            d_cat_str = ", ".join(f"{k}:{v}" for k, v in sorted(r["docling_categories"].items()))
            print(f"  {r['pdf']} p{r['page']}: GT={r['ground_truth']}, "
                  f"Docling={r['docling']} — {direction} [{d_cat_str}]")

    # ── Success criteria check ──
    print(f"\n{'='*70}")
    print("SUCCESS CRITERIA CHECK")
    print(f"{'='*70}")

    d_binary_acc = 100 * d_proc_correct / total
    print(f"Binary should_process accuracy: {d_binary_acc:.1f}% (target: >= 90%)")
    print(f"  {'PASS' if d_binary_acc >= 90 else 'FAIL'}")

    print(f"Speed: {d_avg:.0f} ms/page (target: < 500ms/page)")
    print(f"  {'PASS' if d_avg < 500 else 'FAIL'}")

    # Check for systematic failures
    for cat in ["structures", "table"]:
        cat_results = [r for r in results if r["gt_normalized"] == cat]
        if cat_results:
            cat_correct = sum(1 for r in cat_results if r["d_should_process"])
            cat_recall = 100 * cat_correct / len(cat_results)
            print(f"Recall for {cat}: {cat_recall:.0f}% ({cat_correct}/{len(cat_results)})")
            if cat_recall < 50:
                print(f"  FAIL — systematic failure on {cat} pages")
            else:
                print(f"  PASS")

    print("\nDone.")


if __name__ == "__main__":
    main()
