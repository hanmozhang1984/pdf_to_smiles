#!/usr/bin/env python3
"""Evaluation harness for compound classifier accuracy.

Compares classifier output (which structures are example_compound vs other)
against ground truth Excel files to compute precision, recall, and F1.

Usage:
    python scripts/evaluate_classifier.py \\
        --pdf path/to/patent.pdf \\
        --ground-truth path/to/verified.xlsx \\
        --mode ollama          # or claude, none
        [--ollama-model qwen3.5:9b]
        [--pages 50-100]       # optional page range
        [--verbose]

The ground truth Excel must have a 'Compound_ID' column listing verified
example compound IDs. The evaluator:
  1. Renders each page of the PDF
  2. Runs structure detection (using the configured inference backend)
  3. Runs the classifier (claude / ollama / none)
  4. Compares classifier-assigned compound IDs against ground truth
  5. Reports per-page and overall precision / recall / F1
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

# Add project src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def parse_page_range(page_range_str: str, max_page: int) -> List[int]:
    """Parse a page range string like '1-5,8,12-15' into a list of page numbers."""
    pages = []
    for part in page_range_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            pages.extend(range(int(start), int(end) + 1))
        else:
            pages.append(int(part))
    return [p for p in pages if 1 <= p <= max_page]


def load_ground_truth(excel_path: str) -> Set[str]:
    """Load verified compound IDs from ground truth Excel.

    Returns a set of string compound IDs (normalized).
    """
    import pandas as pd

    df = pd.read_excel(excel_path)

    # Find the compound ID column
    id_col = None
    for col in df.columns:
        if "compound" in col.lower() and "id" in col.lower():
            id_col = col
            break
        if col.lower() in ("compound_id", "compoundid", "id", "example"):
            id_col = col
            break

    if id_col is None:
        # Fall back to first column
        id_col = df.columns[0]
        print(f"Warning: No 'Compound_ID' column found, using '{id_col}'")

    ids = set()
    for val in df[id_col].dropna():
        normalized = str(val).strip()
        # Remove trailing .0 from float-parsed integers
        if normalized.endswith(".0"):
            normalized = normalized[:-2]
        ids.add(normalized)

    return ids


def run_evaluation(
    pdf_path: str,
    ground_truth_ids: Set[str],
    classifier_mode: str,
    ollama_model: str = "qwen3.5:9b",
    prompt_path: Optional[str] = None,
    pages: Optional[List[int]] = None,
    verbose: bool = False,
) -> Dict:
    """Run the classifier on a patent and compare against ground truth.

    Returns a dict with evaluation metrics.
    """
    import pypdfium2 as pdfium
    from PIL import Image

    # Configure classifier mode in settings
    from pdf_to_smiles.core.inference_settings import (
        InferenceSettings,
        ClassifierMode,
    )

    settings = InferenceSettings.get_instance()
    settings.apply_api_keys()

    mode_map = {
        "claude": ClassifierMode.CLAUDE,
        "ollama": ClassifierMode.OLLAMA,
        "none": ClassifierMode.NONE,
    }
    settings._classifier_mode = mode_map.get(classifier_mode, ClassifierMode.CLAUDE)
    if classifier_mode == "ollama":
        settings._ollama_model = ollama_model

    # Set up classifier directly
    classifier = None
    if classifier_mode == "ollama":
        from pdf_to_smiles.core.ollama_compound_classifier import (
            OllamaCompoundClassifier,
            is_available,
        )

        if not is_available(ollama_model):
            print(f"Error: Ollama not available with model '{ollama_model}'")
            sys.exit(1)
        classifier = OllamaCompoundClassifier(
            model=ollama_model, prompt_path=prompt_path
        )
    elif classifier_mode == "claude":
        from pdf_to_smiles.core.llm_compound_classifier import (
            LLMCompoundClassifier,
            is_available,
        )

        if not is_available():
            print("Error: Claude classifier not available (check ANTHROPIC_API_KEY)")
            sys.exit(1)
        classifier = LLMCompoundClassifier()

    # Open PDF
    doc = pdfium.PdfDocument(pdf_path)
    total_pages = len(doc)
    print(f"PDF: {os.path.basename(pdf_path)} ({total_pages} pages)")
    print(f"Ground truth: {len(ground_truth_ids)} compound IDs")
    print(f"Classifier mode: {classifier_mode}")
    print()

    if pages is None:
        pages = list(range(1, total_pages + 1))

    # We need structure detection — use the inference provider
    from pdf_to_smiles.core.inference_provider import InferenceProvider

    provider = InferenceProvider()

    # Track results
    all_predicted_example_ids: Set[str] = set()
    all_predicted_other_ids: Set[str] = set()
    page_results = []
    total_structures = 0
    total_example_compound = 0
    total_other = 0
    misclassifications = []

    for page_num in pages:
        if page_num > total_pages:
            continue

        # Render page
        page = doc[page_num - 1]
        scale = 2.0  # 144 DPI
        bitmap = page.render(scale=scale)
        pil_image = bitmap.to_pil()

        # Detect structures
        try:
            detection_result = provider.detect_structures(pil_image)
        except Exception as e:
            if verbose:
                print(f"  Page {page_num}: structure detection failed: {e}")
            continue

        if not detection_result or not detection_result.get("structures"):
            continue

        structures = detection_result["structures"]
        n_structures = len(structures)

        # Extract bounding boxes
        boxes = []
        for s in structures:
            box = s.get("bbox") or s.get("box")
            if box and len(box) == 4:
                boxes.append(tuple(box))
            else:
                boxes.append(None)

        # If we have no valid boxes, skip classification
        valid_boxes = [b for b in boxes if b is not None]
        if not valid_boxes:
            if verbose:
                print(f"  Page {page_num}: {n_structures} structures, no boxes")
            continue

        # Fill None boxes with center-of-page fallback
        pw, ph = pil_image.size
        boxes = [
            b if b is not None else (pw // 4, ph // 4, 3 * pw // 4, 3 * ph // 4)
            for b in boxes
        ]

        # Classify
        if classifier is None:
            # No classifier — treat all as example_compound
            classifications = [{"type": "example_compound", "id": None}] * n_structures
        else:
            try:
                classifications = classifier.classify_page_structures(
                    pil_image, boxes, page_num=page_num
                )
            except Exception as e:
                if verbose:
                    print(f"  Page {page_num}: classification failed: {e}")
                classifications = [{"type": "other", "id": None}] * n_structures

        # Tally
        page_examples = 0
        page_others = 0
        for cls in classifications:
            total_structures += 1
            if cls["type"] == "example_compound":
                total_example_compound += 1
                page_examples += 1
                if cls.get("id"):
                    all_predicted_example_ids.add(str(cls["id"]))
            else:
                total_other += 1
                page_others += 1
                if cls.get("id"):
                    all_predicted_other_ids.add(str(cls["id"]))

        if verbose:
            print(
                f"  Page {page_num}: {n_structures} structures → "
                f"{page_examples} example, {page_others} other"
            )
            for i, cls in enumerate(classifications):
                cid = cls.get("id", "-")
                ctype = cls["type"]
                marker = ""
                if cid and cid in ground_truth_ids and ctype == "other":
                    marker = " *** FALSE NEGATIVE"
                    misclassifications.append(
                        f"  Page {page_num}, box {i+1}: ID={cid} classified as "
                        f"'other' but is in ground truth"
                    )
                elif cid and cid not in ground_truth_ids and ctype == "example_compound":
                    marker = " *** FALSE POSITIVE"
                    misclassifications.append(
                        f"  Page {page_num}, box {i+1}: ID={cid} classified as "
                        f"'example_compound' but not in ground truth"
                    )
                if verbose and (marker or cid):
                    print(f"    Box {i+1}: {ctype} id={cid}{marker}")

        page_results.append(
            {
                "page": page_num,
                "n_structures": n_structures,
                "n_example": page_examples,
                "n_other": page_others,
            }
        )

    provider.close()

    # Compute metrics based on compound IDs
    # True positives: IDs predicted as example_compound that are in ground truth
    tp = len(all_predicted_example_ids & ground_truth_ids)
    # False positives: IDs predicted as example_compound NOT in ground truth
    fp = len(all_predicted_example_ids - ground_truth_ids)
    # False negatives: ground truth IDs classified as 'other' or missed entirely
    fn_classified_other = len(all_predicted_other_ids & ground_truth_ids)
    # IDs in ground truth but never seen by classifier
    fn_missed = len(ground_truth_ids - all_predicted_example_ids - all_predicted_other_ids)
    fn = fn_classified_other + fn_missed

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    results = {
        "pdf": os.path.basename(pdf_path),
        "mode": classifier_mode,
        "total_pages_processed": len(page_results),
        "total_structures": total_structures,
        "total_example_compound": total_example_compound,
        "total_other": total_other,
        "ground_truth_count": len(ground_truth_ids),
        "predicted_example_ids": len(all_predicted_example_ids),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "fn_classified_other": fn_classified_other,
        "fn_missed": fn_missed,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Pages processed:     {results['total_pages_processed']}")
    print(f"Total structures:    {results['total_structures']}")
    print(f"  Example compound:  {results['total_example_compound']}")
    print(f"  Other:             {results['total_other']}")
    print()
    print(f"Ground truth IDs:    {results['ground_truth_count']}")
    print(f"Predicted example:   {results['predicted_example_ids']}")
    print()
    print(f"True positives:      {tp}")
    print(f"False positives:     {fp}")
    print(f"False negatives:     {fn} ({fn_classified_other} misclassified + {fn_missed} missed)")
    print()
    print(f"Precision:           {precision:.4f}")
    print(f"Recall:              {recall:.4f}")
    print(f"F1 Score:            {f1:.4f}")
    print("=" * 60)

    if misclassifications:
        print("\nMisclassifications:")
        for m in misclassifications:
            print(m)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate compound classifier accuracy against ground truth"
    )
    parser.add_argument("--pdf", required=True, help="Path to patent PDF")
    parser.add_argument(
        "--ground-truth", required=True, help="Path to ground truth Excel file"
    )
    parser.add_argument(
        "--mode",
        choices=["claude", "ollama", "none"],
        default="ollama",
        help="Classifier mode (default: ollama)",
    )
    parser.add_argument(
        "--ollama-model",
        default="qwen3.5:9b",
        help="Ollama model name (default: qwen3.5:9b)",
    )
    parser.add_argument(
        "--prompt-path",
        default=None,
        help="Path to optimized prompt file",
    )
    parser.add_argument(
        "--pages",
        default=None,
        help="Page range to evaluate (e.g., '50-100')",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Save results JSON to this path",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    # Load ground truth
    ground_truth_ids = load_ground_truth(args.ground_truth)
    print(f"Loaded {len(ground_truth_ids)} compound IDs from ground truth")

    # Parse page range
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(args.pdf)
    total_pages = len(doc)
    del doc

    pages = None
    if args.pages:
        pages = parse_page_range(args.pages, total_pages)
        print(f"Evaluating pages: {pages[0]}-{pages[-1]} ({len(pages)} pages)")

    # Run evaluation
    start_time = time.time()
    results = run_evaluation(
        pdf_path=args.pdf,
        ground_truth_ids=ground_truth_ids,
        classifier_mode=args.mode,
        ollama_model=args.ollama_model,
        prompt_path=args.prompt_path,
        pages=pages,
        verbose=args.verbose,
    )
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s")

    # Save results if requested
    if args.output:
        results["elapsed_seconds"] = round(elapsed, 1)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
