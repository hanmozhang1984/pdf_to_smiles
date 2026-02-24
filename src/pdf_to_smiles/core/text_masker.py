"""Detect and mask non-structural text on cropped chemical structure images.

Uses PaddleOCR to find text regions, then classifies each as either a
legitimate atom label (keep) or contaminating text like compound names,
captions, step labels (mask with white fill).

Requires optional dependency: pip install paddleocr paddlepaddle
"""

from __future__ import annotations

import re

import cv2
import numpy as np
from PIL import Image

# Lazy-loaded PaddleOCR instance
_ocr_engine = None

# Pattern matching atom labels and common chemical group abbreviations.
# These should NOT be masked even though OCR detects them as text.
_ATOM_LABEL_RE = re.compile(
    r'^[+-]$'                           # charge symbols
    r'|^[A-Z][a-z]?[0-9]?[+-]?$'       # single atoms optionally numbered/charged: O, N, Br, C3, N+
    r'|^[RCXYA]\d?[a-z]?$'             # R-groups: R, R1, R2, R1a, X, Y
    r"|^R['\u2032]$"                    # R', R′
    r'|^(OH|HO|NH|HN|NH2|NO2|CN|CO|SH|HS)$'  # common 2-3 char groups (incl. reversed)
    r'|^(Me|Et|Ph|Ac|Bz|Bn|Ts)$'       # abbreviations
    r'|^(Boc|Fmoc|Cbz|TBS|TMS|TIPS)$'  # protecting groups
    r'|^(CH[23]|CF3|SO2|PO4|CO2)$'     # subscript groups
    r'|^[nmi]$'                         # polymer repeat / generic indices
)


def is_available() -> bool:
    """Check if PaddleOCR is installed and usable."""
    try:
        import paddleocr  # noqa: F401
        return True
    except ImportError:
        return False


def _get_ocr():
    """Return a lazily-initialized PaddleOCR engine."""
    global _ocr_engine
    if _ocr_engine is None:
        from paddleocr import PaddleOCR
        _ocr_engine = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
            lang='en',
        )
    return _ocr_engine


def _is_atom_label(text: str, box_width: int, img_width: int) -> bool:
    """Return True if *text* looks like an atom label rather than contamination."""
    text = text.strip()

    # Strip trailing hyphens/dashes that OCR picks up from adjacent bond lines
    text = text.rstrip('-–—')

    # Very short text that matches chemical patterns → keep
    if len(text) <= 3 and _ATOM_LABEL_RE.match(text):
        return True

    # Even if regex didn't match, very short text in a small box is likely
    # an atom label (handles OCR mis-reads like "0" for "O")
    if len(text) <= 2 and box_width < img_width * 0.10:
        return True

    return False


def mask_text_regions(image: Image.Image) -> Image.Image:
    """Detect and white-fill non-structural text on a structure image.

    Atom labels (OH, Br, CH3, R1 …) are preserved.  Compound names,
    captions, step labels, and other contaminating text are masked.

    Args:
        image: PIL Image of a cropped chemical structure (RGB recommended).

    Returns:
        A copy of *image* with contaminating text regions white-filled.
        Returns the original image unchanged when PaddleOCR is not installed
        or when masking would remove too much ink (>40% of total).
    """
    if not is_available():
        return image

    if image.mode != 'RGB':
        image = image.convert('RGB')

    img_array = np.array(image)
    img_h, img_w = img_array.shape[:2]

    ocr = _get_ocr()
    results = list(ocr.predict(img_array))

    if not results:
        return image

    result = results[0]
    rec_texts = result.get('rec_texts', [])
    rec_polys = result.get('dt_polys', [])

    if not rec_texts:
        return image

    # Collect boxes to mask
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    padding = 3  # px around each text box

    for text, poly in zip(rec_texts, rec_polys):
        # poly is an array of quad points [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        poly = np.array(poly)
        xs = poly[:, 0]
        ys = poly[:, 1]
        box_w = float(xs.max() - xs.min())

        if _is_atom_label(text, int(box_w), img_w):
            continue

        # Build a padded axis-aligned rectangle for white-filling
        x_min = max(int(xs.min()) - padding, 0)
        y_min = max(int(ys.min()) - padding, 0)
        x_max = min(int(xs.max()) + padding, img_w)
        y_max = min(int(ys.max()) + padding, img_h)
        mask[y_min:y_max, x_min:x_max] = 255

    if cv2.countNonZero(mask) == 0:
        return image

    # Safety: if masking would remove >40% of total ink, skip
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    total_ink = cv2.countNonZero(binary)
    ink_in_mask = cv2.countNonZero(binary & mask)

    if total_ink > 0 and ink_in_mask / total_ink > 0.40:
        return image

    # White-fill masked regions
    cleaned = img_array.copy()
    cleaned[mask == 255] = [255, 255, 255]

    return Image.fromarray(cleaned)
