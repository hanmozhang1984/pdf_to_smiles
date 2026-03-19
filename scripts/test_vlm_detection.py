#!/usr/bin/env python3
"""Prototype: VLM-based unified detection + classification.

Sends full page images to Qwen3-VL and asks it to return bounding boxes,
classification, and compound IDs in a single call — replacing the separate
Docling detection + VLM classification pipeline.

Usage:
    python scripts/test_vlm_detection.py --pdf path/to/patent.pdf --pages 45,48,88,97,99,100
    python scripts/test_vlm_detection.py --pdf path/to/patent.pdf --pages 44-100 --labels training_data/GLP/labels.json
"""

import argparse
import base64
import io
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

import pypdfium2 as pdfium
import requests
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_IMAGE_DIM = 1024  # Max dimension for VLM input
_RENDER_DPI = 200
_REQUEST_TIMEOUT = 300  # seconds
_DEFAULT_BASE_URL = "http://localhost:8000"
_DEFAULT_MODEL = "mlx-community/Qwen3-VL-8B-Instruct-4bit"

_DETECTION_PROMPT = """\
You are analyzing a pharmaceutical patent page. Find ALL chemical structures on this page.

For each structure, return:
- "bbox": [x1, y1, x2, y2] — bounding box in normalized coordinates [0-1000]
- "type": "example_compound" or "other"
- "id": the Example/Compound number (e.g., "25") or null

Rules for classification:
- Table rows with Example numbers + structures → "example_compound"
- Final product of a synthesis scheme with "Example X" label → "example_compound"
- Intermediates, reagents, starting materials, Markush/generic → "other"

IMPORTANT: Draw tight boxes around INDIVIDUAL structures, not entire reaction schemes.
For synthesis schemes, box only the final product separately.

Respond ONLY with JSON:
{"structures": [
  {"bbox": [x1, y1, x2, y2], "type": "example_compound", "id": "25"},
  {"bbox": [x1, y1, x2, y2], "type": "other", "id": null}
]}

If there are no chemical structures on this page, respond with:
{"structures": []}
"""

# ---------------------------------------------------------------------------
# Utility helpers (adapted from ollama_compound_classifier / mlx_compound_classifier)
# ---------------------------------------------------------------------------


def _encode_image(pil_image: Image.Image) -> str:
    """Base64-encode a PIL image as PNG."""
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def _resize_for_api(image: Image.Image) -> Image.Image:
    """Resize image if longer edge exceeds the optimal limit."""
    w, h = image.size
    if max(w, h) <= _MAX_IMAGE_DIM:
        return image
    scale = _MAX_IMAGE_DIM / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return image.resize((new_w, new_h), Image.LANCZOS)


def _extract_json(text: str) -> str:
    """Extract JSON from model output, handling markdown code blocks."""
    code_block_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if code_block_match:
        return code_block_match.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------


def render_page(pdf_path: str, page_num: int, dpi: int = _RENDER_DPI) -> Image.Image:
    """Render a single PDF page to PIL Image."""
    doc = pdfium.PdfDocument(pdf_path)
    page = doc[page_num - 1]  # 0-indexed
    scale = dpi / 72.0
    bitmap = page.render(scale=scale)
    pil_image = bitmap.to_pil()
    doc.close()
    return pil_image


def call_vlm_detection(
    image: Image.Image,
    base_url: str = _DEFAULT_BASE_URL,
    model: str = _DEFAULT_MODEL,
) -> list[dict]:
    """Send page image to Qwen3-VL and parse detection results.

    Returns list of dicts with keys: bbox (pixel coords), type, id.
    """
    orig_w, orig_h = image.size
    resized = _resize_for_api(image)
    resized_w, resized_h = resized.size
    image_b64 = _encode_image(resized)

    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _DETECTION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 2048,
            "temperature": 0,
        },
        timeout=_REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    raw_text = response.json()["choices"][0]["message"]["content"].strip()
    # Strip <think>...</think> blocks from reasoning models
    raw_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()

    json_str = _extract_json(raw_text)
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        print(f"  [WARN] Failed to parse JSON from VLM response:\n{raw_text[:500]}")
        return []

    structures = data.get("structures", [])

    # Convert [0, 1000] normalized coords → pixel coords on original image
    results = []
    for s in structures:
        bbox_norm = s.get("bbox", [0, 0, 0, 0])
        if len(bbox_norm) != 4:
            continue
        x1 = int(bbox_norm[0] / 1000 * orig_w)
        y1 = int(bbox_norm[1] / 1000 * orig_h)
        x2 = int(bbox_norm[2] / 1000 * orig_w)
        y2 = int(bbox_norm[3] / 1000 * orig_h)
        results.append(
            {
                "bbox": [x1, y1, x2, y2],
                "type": s.get("type", "other"),
                "id": s.get("id"),
            }
        )
    return results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def annotate_image(
    image: Image.Image,
    detections: list[dict],
    gt_structures: Optional[dict] = None,
) -> Image.Image:
    """Draw detection boxes on the image. Optionally overlay ground truth."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    # Try to use a monospace font; fall back to default
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 16)
    except (OSError, IOError):
        font = ImageFont.load_default()

    # Draw ground truth boxes first (blue, dashed-style via thinner lines)
    if gt_structures:
        for sid, gt in gt_structures.items():
            bx = gt["bbox"]
            draw.rectangle(bx, outline="blue", width=2)
            label = f"GT:{gt['type'][:3]}"
            if gt.get("id"):
                label += f" #{gt['id']}"
            draw.text((bx[0], bx[1] - 18), label, fill="blue", font=font)

    # Draw VLM detections
    for det in detections:
        bx = det["bbox"]
        color = "lime" if det["type"] == "example_compound" else "gray"
        draw.rectangle(bx, outline=color, width=3)
        label = det["type"][:3]
        if det.get("id"):
            label += f" #{det['id']}"
        draw.text((bx[0], bx[1] - 18), label, fill=color, font=font)

    return annotated


# ---------------------------------------------------------------------------
# Evaluation against ground truth
# ---------------------------------------------------------------------------


def compute_iou(box_a: list, box_b: list) -> float:
    """Compute intersection-over-union of two [x1,y1,x2,y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def evaluate_page(
    detections: list[dict],
    gt_structures: dict,
    iou_threshold: float = 0.3,
) -> dict:
    """Compare VLM detections against ground truth for one page.

    Returns dict with per-page metrics.
    """
    gt_list = list(gt_structures.values())
    matched_gt = set()
    matched_det = set()
    type_correct = 0
    id_correct = 0
    ious = []

    # Match each GT structure to the best overlapping detection
    for gi, gt in enumerate(gt_list):
        best_iou = 0.0
        best_di = -1
        for di, det in enumerate(detections):
            if di in matched_det:
                continue
            iou = compute_iou(gt["bbox"], det["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_di = di
        if best_iou >= iou_threshold and best_di >= 0:
            matched_gt.add(gi)
            matched_det.add(best_di)
            ious.append(best_iou)
            det = detections[best_di]
            if det["type"] == gt["type"]:
                type_correct += 1
            # Normalize IDs for comparison
            gt_id = str(gt["id"]).strip() if gt.get("id") else None
            det_id = str(det["id"]).strip() if det.get("id") else None
            if gt_id == det_id:
                id_correct += 1

    n_gt = len(gt_list)
    n_det = len(detections)
    n_matched = len(matched_gt)
    n_gt_examples = sum(1 for g in gt_list if g["type"] == "example_compound")
    n_det_examples = sum(1 for d in detections if d["type"] == "example_compound")

    return {
        "n_gt": n_gt,
        "n_detected": n_det,
        "n_matched": n_matched,
        "recall": n_matched / n_gt if n_gt else 1.0,
        "precision": n_matched / n_det if n_det else 1.0,
        "type_accuracy": type_correct / n_matched if n_matched else 0.0,
        "id_accuracy": id_correct / n_matched if n_matched else 0.0,
        "mean_iou": sum(ious) / len(ious) if ious else 0.0,
        "false_positives": n_det - n_matched,
        "missed": n_gt - n_matched,
        "gt_examples": n_gt_examples,
        "det_examples": n_det_examples,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_pages(pages_str: str) -> list[int]:
    """Parse page spec like '45,48,88-100' into sorted list of ints."""
    pages = set()
    for part in pages_str.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            pages.update(range(int(lo), int(hi) + 1))
        else:
            pages.add(int(part))
    return sorted(pages)


def main():
    parser = argparse.ArgumentParser(
        description="Prototype VLM-based structure detection"
    )
    parser.add_argument("--pdf", required=True, help="Path to patent PDF")
    parser.add_argument(
        "--pages",
        required=True,
        help="Page numbers: e.g. '45,48,88-100'",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help="Path to labels.json ground truth (optional)",
    )
    parser.add_argument(
        "--output-dir",
        default="output/vlm_detection",
        help="Directory for annotated images + results",
    )
    parser.add_argument("--base-url", default=_DEFAULT_BASE_URL)
    parser.add_argument("--model", default=_DEFAULT_MODEL)
    parser.add_argument("--dpi", type=int, default=_RENDER_DPI)
    args = parser.parse_args()

    pages = parse_pages(args.pages)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load ground truth if provided
    gt_data = None
    if args.labels:
        gt_data = json.loads(Path(args.labels).read_text())

    all_results = {}
    all_metrics = {}
    total_time = 0.0

    print(f"Processing {len(pages)} pages from {args.pdf}")
    print(f"VLM endpoint: {args.base_url} / {args.model}")
    print(f"Output: {out_dir}\n")

    for page_num in pages:
        print(f"--- Page {page_num} ---")

        # 1. Render page
        pil_image = render_page(args.pdf, page_num, dpi=args.dpi)
        print(f"  Rendered: {pil_image.size[0]}x{pil_image.size[1]} px")

        # 2. Call VLM
        t0 = time.time()
        detections = call_vlm_detection(
            pil_image, base_url=args.base_url, model=args.model
        )
        elapsed = time.time() - t0
        total_time += elapsed
        print(f"  VLM: {len(detections)} structures detected ({elapsed:.1f}s)")

        for det in detections:
            print(f"    {det['type']:20s}  id={det['id']!s:>5s}  bbox={det['bbox']}")

        # 3. Get ground truth for this page
        gt_structs = None
        if gt_data and str(page_num) in gt_data:
            gt_structs = gt_data[str(page_num)]["structures"]

        # 4. Annotate and save image
        annotated = annotate_image(pil_image, detections, gt_structs)
        img_path = out_dir / f"page_{page_num:03d}.png"
        annotated.save(str(img_path))
        print(f"  Saved: {img_path}")

        # 5. Evaluate if ground truth available
        if gt_structs:
            metrics = evaluate_page(detections, gt_structs)
            all_metrics[page_num] = metrics
            print(
                f"  Eval: recall={metrics['recall']:.0%} "
                f"precision={metrics['precision']:.0%} "
                f"type_acc={metrics['type_accuracy']:.0%} "
                f"id_acc={metrics['id_accuracy']:.0%} "
                f"mean_iou={metrics['mean_iou']:.2f} "
                f"FP={metrics['false_positives']} missed={metrics['missed']}"
            )

        all_results[page_num] = detections
        print()

    # Save raw results JSON
    results_path = out_dir / "results.json"
    results_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"Results saved to {results_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY — {len(pages)} pages, {total_time:.1f}s total")
    print(f"{'='*60}")

    total_detected = sum(len(d) for d in all_results.values())
    print(f"Total structures detected: {total_detected}")

    if all_metrics:
        total_gt = sum(m["n_gt"] for m in all_metrics.values())
        total_matched = sum(m["n_matched"] for m in all_metrics.values())
        total_fp = sum(m["false_positives"] for m in all_metrics.values())
        total_missed = sum(m["missed"] for m in all_metrics.values())
        total_type_ok = sum(
            m["type_accuracy"] * m["n_matched"] for m in all_metrics.values()
        )
        total_id_ok = sum(
            m["id_accuracy"] * m["n_matched"] for m in all_metrics.values()
        )

        overall_recall = total_matched / total_gt if total_gt else 0
        overall_precision = total_matched / (total_matched + total_fp) if (total_matched + total_fp) else 0
        overall_type_acc = total_type_ok / total_matched if total_matched else 0
        overall_id_acc = total_id_ok / total_matched if total_matched else 0
        mean_ious = [m["mean_iou"] for m in all_metrics.values() if m["mean_iou"] > 0]
        overall_iou = sum(mean_ious) / len(mean_ious) if mean_ious else 0

        print(f"Ground truth structures: {total_gt}")
        print(f"Matched:    {total_matched}/{total_gt} (recall={overall_recall:.0%})")
        print(f"Precision:  {overall_precision:.0%}")
        print(f"Type acc:   {overall_type_acc:.0%}")
        print(f"ID acc:     {overall_id_acc:.0%}")
        print(f"Mean IoU:   {overall_iou:.2f}")
        print(f"False pos:  {total_fp}")
        print(f"Missed:     {total_missed}")

        # Per-page summary table
        print(f"\n{'Page':>6}  {'GT':>3}  {'Det':>3}  {'Match':>5}  {'Recall':>7}  {'FP':>3}  {'Miss':>4}")
        print("-" * 50)
        for pg in sorted(all_metrics.keys()):
            m = all_metrics[pg]
            print(
                f"{pg:>6}  {m['n_gt']:>3}  {m['n_detected']:>3}  "
                f"{m['n_matched']:>5}  {m['recall']:>6.0%}  "
                f"{m['false_positives']:>3}  {m['missed']:>4}"
            )

    # Save metrics JSON
    if all_metrics:
        metrics_path = out_dir / "metrics.json"
        metrics_path.write_text(json.dumps(all_metrics, indent=2, default=str))
        print(f"\nMetrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
