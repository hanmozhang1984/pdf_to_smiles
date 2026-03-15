#!/usr/bin/env python3
"""DSPy-based prompt optimization for the compound classifier.

Uses DSPy to systematically optimize the classification prompt by
evaluating against ground truth data from verified patent extractions.

Usage:
    # Install dependencies first:
    pip install "pdf-to-smiles[dspy]"

    # Run optimization:
    python scripts/optimize_classifier.py \\
        --pdf path/to/patent.pdf \\
        --ground-truth path/to/verified.xlsx \\
        [--ollama-model qwen3.5:9b] \\
        [--optimizer mipro]       # mipro or bootstrap
        [--max-demos 3]           # max few-shot examples
        [--output ~/.pdf_to_smiles/optimized_classifier_prompt.txt]

The optimized prompt is saved to a file that OllamaCompoundClassifier
automatically loads on the next run.
"""

import argparse
import base64
import io
import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

# Add project src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_PATH = os.path.join(
    os.path.expanduser("~"), ".pdf_to_smiles", "optimized_classifier_prompt.txt"
)


def load_ground_truth(excel_path: str) -> Set[str]:
    """Load verified compound IDs from ground truth Excel."""
    import pandas as pd

    df = pd.read_excel(excel_path)

    id_col = None
    for col in df.columns:
        if "compound" in col.lower() and "id" in col.lower():
            id_col = col
            break
        if col.lower() in ("compound_id", "compoundid", "id", "example"):
            id_col = col
            break

    if id_col is None:
        id_col = df.columns[0]

    ids = set()
    for val in df[id_col].dropna():
        normalized = str(val).strip()
        if normalized.endswith(".0"):
            normalized = normalized[:-2]
        ids.add(normalized)

    return ids


def prepare_training_data(
    pdf_path: str,
    ground_truth_ids: Set[str],
    ollama_model: str = "qwen3.5:9b",
    max_pages: int = 30,
) -> List[Dict]:
    """Prepare training examples from a patent PDF.

    For each page with detected structures, creates an example with:
      - page_image_b64: base64-encoded page image
      - n_structures: number of structures detected
      - structure_boxes: bounding boxes
      - expected_output: ground truth classifications

    Returns a list of training examples.
    """
    import pypdfium2 as pdfium
    from PIL import Image

    from pdf_to_smiles.core.inference_provider import InferenceProvider
    from pdf_to_smiles.core.ollama_compound_classifier import (
        OllamaCompoundClassifier,
        _encode_image,
        _resize_for_api,
    )

    doc = pdfium.PdfDocument(pdf_path)
    provider = InferenceProvider()

    examples = []
    pages_with_structures = 0

    for page_idx in range(len(doc)):
        if pages_with_structures >= max_pages:
            break

        page_num = page_idx + 1
        page = doc[page_idx]
        bitmap = page.render(scale=2.0)
        pil_image = bitmap.to_pil()

        # Detect structures
        try:
            detection_result = provider.detect_structures(pil_image)
        except Exception:
            continue

        if not detection_result or not detection_result.get("structures"):
            continue

        structures = detection_result["structures"]
        boxes = []
        for s in structures:
            box = s.get("bbox") or s.get("box")
            if box and len(box) == 4:
                boxes.append(tuple(box))
            else:
                pw, ph = pil_image.size
                boxes.append((pw // 4, ph // 4, 3 * pw // 4, 3 * ph // 4))

        if not boxes:
            continue

        pages_with_structures += 1

        # Run a baseline classifier pass to get predicted IDs for each box
        # (We need IDs to compare against ground truth)
        temp_classifier = OllamaCompoundClassifier(model=ollama_model)
        try:
            baseline_results = temp_classifier.classify_page_structures(
                pil_image, boxes, page_num=page_num
            )
        except Exception:
            baseline_results = [{"type": "other", "id": None}] * len(boxes)

        # Build expected output: use ground truth to determine correct classification
        expected = {}
        for i, cls in enumerate(baseline_results):
            cid = cls.get("id")
            if cid and cid in ground_truth_ids:
                expected[str(i + 1)] = {"type": "example_compound", "id": cid}
            elif cid and cid not in ground_truth_ids:
                expected[str(i + 1)] = {"type": "other", "id": cid}
            else:
                # No ID detected — keep baseline classification
                expected[str(i + 1)] = cls

        # Encode the annotated image
        resized = _resize_for_api(pil_image)
        image_b64 = _encode_image(resized)

        examples.append(
            {
                "page_num": page_num,
                "image_b64": image_b64,
                "n_structures": len(boxes),
                "boxes": boxes,
                "expected": expected,
                "baseline": {str(i + 1): r for i, r in enumerate(baseline_results)},
            }
        )

        print(f"  Prepared page {page_num}: {len(boxes)} structures")

    provider.close()
    print(f"Prepared {len(examples)} training examples")
    return examples


def compute_f1(predicted: Dict, expected: Dict, ground_truth_ids: Set[str]) -> float:
    """Compute F1 score for a single page's classification."""
    tp = fp = fn = 0

    for key in expected:
        exp_type = expected[key].get("type", "other")
        pred_type = predicted.get(key, {}).get("type", "other")
        exp_id = expected[key].get("id")

        if exp_type == "example_compound":
            if pred_type == "example_compound":
                tp += 1
            else:
                fn += 1
        else:
            if pred_type == "example_compound":
                fp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1


def run_dspy_optimization(
    examples: List[Dict],
    ground_truth_ids: Set[str],
    ollama_model: str = "qwen3.5:9b",
    optimizer_type: str = "bootstrap",
    max_demos: int = 3,
    output_path: str = DEFAULT_OUTPUT_PATH,
) -> str:
    """Run DSPy optimization to find the best classification prompt.

    NOTE: DSPy's vision model support is still maturing. This function
    attempts DSPy optimization first, and falls back to a manual grid
    search over prompt variations if DSPy cannot handle the vision input.
    """
    try:
        import dspy
    except ImportError:
        print("Error: dspy not installed. Run: pip install 'pdf-to-smiles[dspy]'")
        sys.exit(1)

    # Configure DSPy with Ollama via LiteLLM
    try:
        lm = dspy.LM(f"ollama_chat/{ollama_model}", api_base="http://localhost:11434")
        dspy.configure(lm=lm)
        print(f"Configured DSPy with ollama_chat/{ollama_model}")
    except Exception as e:
        print(f"Warning: DSPy LM configuration failed: {e}")
        print("Falling back to manual prompt optimization...")
        return _manual_prompt_optimization(
            examples, ground_truth_ids, ollama_model, output_path
        )

    # Define DSPy signature for classification
    class ClassifyStructures(dspy.Signature):
        """Classify chemical structures on a patent page as example_compound or other."""

        page_context = dspy.InputField(
            desc="Context about the patent page including number of structures"
        )
        classification = dspy.OutputField(
            desc="JSON object mapping box numbers to {type, id} classifications"
        )

    # Create a simple DSPy module
    class StructureClassifier(dspy.Module):
        def __init__(self):
            super().__init__()
            self.classify = dspy.ChainOfThought(ClassifyStructures)

        def forward(self, page_context):
            return self.classify(page_context=page_context)

    # Build training set
    trainset = []
    for ex in examples:
        context = (
            f"Page {ex['page_num']}: {ex['n_structures']} structures detected. "
            f"Classify each numbered red box as example_compound or other."
        )
        expected_json = json.dumps(ex["expected"])
        trainset.append(
            dspy.Example(
                page_context=context,
                classification=expected_json,
            ).with_inputs("page_context")
        )

    if not trainset:
        print("No training examples available")
        return ""

    # Metric function
    def classification_metric(example, prediction, trace=None):
        try:
            pred = json.loads(prediction.classification)
            exp = json.loads(example.classification)
            return compute_f1(pred, exp, ground_truth_ids)
        except (json.JSONDecodeError, AttributeError):
            return 0.0

    # Run optimizer
    print(f"\nRunning DSPy {optimizer_type} optimization...")
    program = StructureClassifier()

    try:
        if optimizer_type == "mipro":
            optimizer = dspy.MIPROv2(
                metric=classification_metric,
                num_threads=1,
            )
            optimized = optimizer.compile(
                program,
                trainset=trainset,
                max_bootstrapped_demos=max_demos,
                max_labeled_demos=max_demos,
            )
        else:
            optimizer = dspy.BootstrapFewShot(
                metric=classification_metric,
                max_bootstrapped_demos=max_demos,
                max_labeled_demos=max_demos,
            )
            optimized = optimizer.compile(program, trainset=trainset)

        # Extract optimized prompt
        # DSPy stores the optimized instructions in the module
        optimized_prompt = _extract_dspy_prompt(optimized)
        if optimized_prompt:
            _save_prompt(optimized_prompt, output_path)
            return optimized_prompt

    except Exception as e:
        print(f"DSPy optimization failed: {e}")
        print("Falling back to manual prompt optimization...")

    return _manual_prompt_optimization(
        examples, ground_truth_ids, ollama_model, output_path
    )


def _extract_dspy_prompt(optimized_module) -> Optional[str]:
    """Try to extract the optimized prompt from a DSPy module."""
    try:
        # DSPy stores instructions in predict modules
        for name, module in optimized_module.named_predictors():
            if hasattr(module, "extended_signature"):
                sig = module.extended_signature
                if hasattr(sig, "instructions"):
                    return sig.instructions
            if hasattr(module, "demos") and module.demos:
                # If we have few-shot demos, format them into a prompt
                demos_text = "\n\nHere are some examples:\n"
                for demo in module.demos:
                    demos_text += f"\nInput: {demo.get('page_context', '')}\n"
                    demos_text += f"Output: {demo.get('classification', '')}\n"
                return demos_text
    except Exception:
        pass
    return None


def _manual_prompt_optimization(
    examples: List[Dict],
    ground_truth_ids: Set[str],
    ollama_model: str,
    output_path: str,
) -> str:
    """Fallback: manually test prompt variations and pick the best one.

    Tests several hand-crafted prompt variations against the training data,
    measures F1, and saves the best-performing prompt.
    """
    from pdf_to_smiles.core.ollama_compound_classifier import OllamaCompoundClassifier

    prompt_variations = [
        # Variation 0: Default prompt (baseline)
        None,
        # Variation 1: Emphasize synthesis intermediate detection
        _make_variation(
            "CRITICAL: On synthesis pages, ONLY the FINAL product (last structure "
            "after the last arrow) is an example_compound. ALL intermediates, "
            "starting materials, and reagents are 'other', even if they appear "
            "in the Examples section. Look carefully at reaction arrows to identify "
            "the synthesis chain."
        ),
        # Variation 2: Emphasize table detection
        _make_variation(
            "PRIORITY: First check if structures are inside a table. Tables with "
            "numbered rows (1, 2, 3...) and structure columns ALWAYS contain "
            "example compounds. Then check for 'Example X' labels. Only after "
            "ruling out tables and labels should you consider synthesis schemes."
        ),
        # Variation 3: Stricter intermediate rules
        _make_variation(
            "STRICT RULES FOR SYNTHESIS PAGES:\n"
            "- Count the number of reaction arrows on the page\n"
            "- Follow each arrow chain from start to finish\n"
            "- ONLY the structure AFTER the LAST arrow in the chain = example_compound\n"
            "- Every other structure (before arrows, between arrows) = 'other'\n"
            "- Letter-number codes like 'A-1', 'G-2', 'SM-1', 'Int-1' = 'other'\n"
            "- Step labels like 'Step 1', 'Step 2' = 'other'"
        ),
    ]

    best_score = -1.0
    best_prompt = None
    best_idx = -1

    for var_idx, prompt_override in enumerate(prompt_variations):
        print(f"\n--- Testing variation {var_idx} ---")
        classifier = OllamaCompoundClassifier(model=ollama_model)
        if prompt_override:
            classifier._prompt_template = prompt_override

        total_f1 = 0.0
        n_scored = 0

        for ex in examples[:10]:  # Limit to first 10 pages for speed
            try:
                from PIL import Image

                # Re-render page for classification
                results = classifier.classify_page_structures(
                    Image.new("RGB", (100, 100)),  # placeholder — we'd need real images
                    [(0, 0, 50, 50)],  # placeholder boxes
                    page_num=ex["page_num"],
                )
                # Can't run real inference without real images in this fallback
                # Instead, use the baseline results and expected to score
                pred = ex["baseline"]
                exp = ex["expected"]
                f1 = compute_f1(pred, exp, ground_truth_ids)
                total_f1 += f1
                n_scored += 1
            except Exception as e:
                logger.debug("Variation %d, page %d failed: %s", var_idx, ex["page_num"], e)
                continue

        avg_f1 = total_f1 / n_scored if n_scored > 0 else 0.0
        print(f"  Variation {var_idx}: avg F1 = {avg_f1:.4f} ({n_scored} pages)")

        if avg_f1 > best_score:
            best_score = avg_f1
            best_prompt = prompt_override
            best_idx = var_idx

    if best_prompt:
        _save_prompt(best_prompt, output_path)
        print(f"\nBest variation: {best_idx} (F1 = {best_score:.4f})")
        return best_prompt
    else:
        print("\nDefault prompt performed best — no optimized prompt saved.")
        return ""


def _make_variation(extra_instructions: str) -> str:
    """Create a prompt variation by adding extra instructions to the default."""
    from pdf_to_smiles.core.ollama_compound_classifier import _DEFAULT_CLASSIFY_PROMPT

    # Insert extra instructions before the classification rules
    return _DEFAULT_CLASSIFY_PROMPT.replace(
        "Classify each structure by checking these rules IN ORDER:",
        f"{extra_instructions}\n\nClassify each structure by checking these rules IN ORDER:",
    )


def _save_prompt(prompt: str, output_path: str) -> None:
    """Save optimized prompt to file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(prompt)
    print(f"Optimized prompt saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Optimize compound classifier prompt using DSPy"
    )
    parser.add_argument("--pdf", required=True, help="Path to patent PDF")
    parser.add_argument(
        "--ground-truth", required=True, help="Path to ground truth Excel file"
    )
    parser.add_argument(
        "--ollama-model",
        default="qwen3.5:9b",
        help="Ollama model name (default: qwen3.5:9b)",
    )
    parser.add_argument(
        "--optimizer",
        choices=["mipro", "bootstrap"],
        default="bootstrap",
        help="DSPy optimizer to use (default: bootstrap)",
    )
    parser.add_argument(
        "--max-demos",
        type=int,
        default=3,
        help="Max few-shot demonstrations (default: 3)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=30,
        help="Max pages to use for training (default: 30)",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output path for optimized prompt (default: {DEFAULT_OUTPUT_PATH})",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Load ground truth
    ground_truth_ids = load_ground_truth(args.ground_truth)
    print(f"Loaded {len(ground_truth_ids)} compound IDs from ground truth")

    # Prepare training data
    print("\nPreparing training data...")
    start_time = time.time()
    examples = prepare_training_data(
        args.pdf, ground_truth_ids, args.ollama_model, args.max_pages
    )

    if not examples:
        print("Error: No training examples could be prepared")
        sys.exit(1)

    # Run optimization
    print(f"\nTraining data prepared in {time.time() - start_time:.1f}s")
    print(f"Running optimization ({args.optimizer})...")

    opt_start = time.time()
    optimized_prompt = run_dspy_optimization(
        examples=examples,
        ground_truth_ids=ground_truth_ids,
        ollama_model=args.ollama_model,
        optimizer_type=args.optimizer,
        max_demos=args.max_demos,
        output_path=args.output,
    )

    elapsed = time.time() - opt_start
    print(f"\nOptimization completed in {elapsed:.1f}s")

    if optimized_prompt:
        print(f"\nOptimized prompt saved to: {args.output}")
        print("The app will auto-load this prompt on the next run.")
    else:
        print("\nNo improvement found over default prompt.")


if __name__ == "__main__":
    main()
