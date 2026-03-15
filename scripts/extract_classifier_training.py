#!/usr/bin/env python3
"""Extract annotated page images for classifier training/labeling.

For each page with detected structures in a patent PDF, this script:
  1. Renders the page at 200 DPI
  2. Detects structures (using the configured detection backend)
  3. Draws numbered red boxes around each structure
  4. Saves the annotated page image as PNG
  5. Creates a labeling template (JSON) for the user to fill in

The user manually labels each red-boxed structure with:
  - type: "example_compound" or "other"
  - id: the compound/example number (or null)

The completed labels can then be used for:
  - DSPy prompt optimization (optimize_classifier.py)
  - Classifier accuracy evaluation (evaluate_classifier.py)

Usage:
    python scripts/extract_classifier_training.py \\
        --pdf path/to/patent.pdf \\
        --output-dir ./training_data/patent_name \\
        [--pages 44-100]          # optional page range
        [--ground-truth gt.xlsx]  # optional: auto-fill labels from GT

    The output directory will contain:
      annotated_pages/
        page_047.png            # annotated page image with red boxes
        page_048.png
        ...
      labels.json               # labeling template (edit this!)
      metadata.json             # run metadata
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


def normalize_id(raw_id: str) -> str:
    """Normalize a compound ID for matching."""
    s = str(raw_id).strip()
    if s.endswith(".0"):
        s = s[:-2]
    s = re.sub(
        r"^(?:Example|Compound|Cpd\.?|Cmpd\.?|Ex\.?|No\.?|#)\s*",
        "", s, flags=re.IGNORECASE,
    ).strip()
    m = re.match(r"^0+(\d+)$", s)
    if m:
        s = m.group(1)
    return s


def load_ground_truth_ids(excel_path: str) -> Set[str]:
    """Load compound IDs from ground truth Excel."""
    try:
        import openpyxl
    except ImportError:
        print("Warning: openpyxl not installed, skipping ground truth")
        return set()

    wb = openpyxl.load_workbook(excel_path, read_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    wb.close()

    if not rows:
        return set()

    header = [str(h).lower().strip() if h else "" for h in rows[0]]

    id_idx = None
    for i, h in enumerate(header):
        if "compound" in h and ("id" in h or "number" in h):
            id_idx = i
            break
        if h in ("compound_id", "id", "example"):
            id_idx = i
            break
    if id_idx is None:
        id_idx = 0

    ids = set()
    for row in rows[1:]:
        if len(row) <= id_idx or row[id_idx] is None:
            continue
        nid = normalize_id(str(row[id_idx]))
        if nid:
            ids.add(nid)

    return ids


def draw_annotated_page(
    page_image: Image.Image,
    boxes: List[Tuple[int, int, int, int]],
) -> Image.Image:
    """Draw numbered red boxes on a page image."""
    annotated = page_image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)

    font = None
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20
            )
        except (OSError, IOError):
            font = ImageFont.load_default()

    for idx, (x1, y1, x2, y2) in enumerate(boxes):
        label = str(idx + 1)
        for offset in range(3):
            draw.rectangle(
                [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                outline="red",
            )
        text_x = x1
        text_y = max(0, y1 - 24)
        bbox = draw.textbbox((text_x, text_y), label, font=font)
        draw.rectangle(
            [bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2],
            fill="white",
        )
        draw.text((text_x, text_y), label, fill="red", font=font)

    return annotated


def parse_page_range(text: str, max_page: int) -> List[int]:
    """Parse '44-100,120' into list of page numbers."""
    pages = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            pages.extend(range(int(a), int(b) + 1))
        elif part:
            pages.append(int(part))
    return [p for p in pages if 1 <= p <= max_page]


def main():
    parser = argparse.ArgumentParser(
        description="Extract annotated page images for classifier labeling"
    )
    parser.add_argument("--pdf", required=True, help="Path to patent PDF")
    parser.add_argument(
        "--output-dir", required=True,
        help="Output directory for annotated images and labels",
    )
    parser.add_argument("--pages", help="Page range (e.g., '44-100')")
    parser.add_argument(
        "--ground-truth",
        help="Optional ground truth Excel to auto-fill known labels",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Render DPI (default: 200)")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    # Load ground truth if provided
    gt_ids = set()
    if args.ground_truth:
        gt_ids = load_ground_truth_ids(args.ground_truth)
        print(f"Loaded {len(gt_ids)} ground-truth compound IDs")

    # Open PDF
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(args.pdf)
    total_pages = len(doc)
    print(f"PDF: {os.path.basename(args.pdf)} ({total_pages} pages)")

    # Determine pages
    if args.pages:
        pages = parse_page_range(args.pages, total_pages)
    else:
        pages = list(range(1, total_pages + 1))

    # Set up detection
    from pdf_to_smiles.core.inference_provider import InferenceProvider
    provider = InferenceProvider()

    # Also try Docling detector if available
    docling_detector = None
    try:
        from pdf_to_smiles.core.page_classifier import get_classifier
        from pdf_to_smiles.core.docling_classifier import DoclingClassifier
        from pdf_to_smiles.core.docling_detector import DoclingDetector

        classifier = get_classifier()
        if isinstance(classifier, DoclingClassifier):
            print("Running Docling page scan for layout detection...")
            classifier.detect_structure_pages(args.pdf)
            docling_detector = DoclingDetector(classifier)
        elif hasattr(classifier, "_primary") and isinstance(
            classifier._primary, DoclingClassifier
        ):
            print("Running Docling page scan (via HybridClassifier)...")
            classifier.detect_structure_pages(args.pdf)
            docling_detector = DoclingDetector(classifier._primary)
    except ImportError:
        pass

    # Create output directory
    img_dir = os.path.join(args.output_dir, "annotated_pages")
    os.makedirs(img_dir, exist_ok=True)

    # Process pages
    labels = {}  # page_num → {box_num → {type, id}}
    metadata = {
        "pdf": os.path.basename(args.pdf),
        "pdf_path": os.path.abspath(args.pdf),
        "total_pages": total_pages,
        "dpi": args.dpi,
        "pages_processed": 0,
        "pages_with_structures": 0,
        "total_structures": 0,
        "ground_truth_file": args.ground_truth,
        "ground_truth_ids": len(gt_ids),
    }

    start = time.time()

    for page_num in pages:
        # Render page
        page = doc[page_num - 1]
        scale = args.dpi / 72.0
        bitmap = page.render(scale=scale)
        pil_image = bitmap.to_pil()

        # Detect structures (try Docling first, then fallback)
        structures_with_boxes = []
        if docling_detector is not None:
            structures_with_boxes = docling_detector.detect_structures_with_boxes(
                pil_image, page_num
            )
        if not structures_with_boxes:
            structures_with_boxes = provider.detect_structures_with_boxes(pil_image)

        if not structures_with_boxes:
            continue

        boxes = [box for _, box in structures_with_boxes if box is not None]
        if not boxes:
            # Try to compute boxes from images
            structure_images = [img for img, _ in structures_with_boxes]
            if structure_images:
                # Use even distribution as fallback
                w, h = pil_image.size
                n = len(structure_images)
                margin = int(h * 0.1)
                usable = h - 2 * margin
                for i, simg in enumerate(structure_images):
                    sw, sh = simg.size
                    y_center = margin + (usable * i // max(n - 1, 1)) if n > 1 else h // 2
                    x1 = max(0, (w - sw) // 2)
                    y1 = max(0, y_center - sh // 2)
                    boxes.append((x1, y1, x1 + sw, y1 + sh))

        if not boxes:
            continue

        metadata["pages_with_structures"] += 1
        metadata["total_structures"] += len(boxes)

        # Draw annotated image
        annotated = draw_annotated_page(pil_image, boxes)
        img_path = os.path.join(img_dir, f"page_{page_num:03d}.png")
        annotated.save(img_path)

        # Create label template for this page
        page_labels = {}
        for i, box in enumerate(boxes):
            box_num = str(i + 1)
            page_labels[box_num] = {
                "type": "TODO",  # User fills: "example_compound" or "other"
                "id": None,       # User fills: compound ID or null
                "bbox": list(box),
            }

        labels[str(page_num)] = {
            "image_file": f"annotated_pages/page_{page_num:03d}.png",
            "n_structures": len(boxes),
            "structures": page_labels,
        }

        if args.verbose:
            print(f"  Page {page_num}: {len(boxes)} structures")

    metadata["pages_processed"] = len(pages)
    doc.close()
    provider.close()

    # If ground truth was provided, auto-fill labels where possible
    # (This is a hint — user should still verify)
    if gt_ids:
        auto_filled = 0
        for page_str, page_data in labels.items():
            for box_num, box_data in page_data["structures"].items():
                # We don't know compound IDs from detection alone,
                # so just mark all as needing review
                pass
        # Note: auto-fill from GT requires compound ID detection,
        # which we don't have here. The user labels manually.
        print(f"\nNote: Ground truth IDs loaded but cannot auto-assign to boxes.")
        print(f"  Use these IDs as reference while labeling: {sorted(gt_ids)[:20]}...")

    # Save labels template
    labels_path = os.path.join(args.output_dir, "labels.json")
    with open(labels_path, "w") as f:
        json.dump(labels, f, indent=2)

    # Save metadata
    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    elapsed = time.time() - start

    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Pages with structures: {metadata['pages_with_structures']}")
    print(f"  Total structures:      {metadata['total_structures']}")
    print(f"\nOutput:")
    print(f"  Annotated images: {img_dir}/")
    print(f"  Labels template:  {labels_path}")
    print(f"  Metadata:         {meta_path}")
    print(f"\nNext steps:")
    print(f"  1. Open each annotated page image in {img_dir}/")
    print(f"  2. Edit {labels_path} — for each numbered red box, set:")
    print(f'     "type": "example_compound" or "other"')
    print(f'     "id": the compound/example number (e.g., "25") or null')
    print(f"  3. Run DSPy optimization:")
    print(f"     python scripts/optimize_classifier.py \\")
    print(f"       --labels {labels_path} \\")
    print(f"       --pdf {args.pdf}")


if __name__ == "__main__":
    main()
