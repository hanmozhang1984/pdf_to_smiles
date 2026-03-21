"""Remove contaminating elements from chemical structure images before OCSR.

Targets: stray lines, partial fragments of neighboring structures, arrows,
and other disconnected elements that bleed into the bounding box crop.
Uses connected component analysis to identify and remove elements that are
spatially disconnected from the main molecular graph.

NOT intended to fix: truncated structures (upstream detection problem),
non-structure images like reaction schemes or tables (P1 classifier problem).
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def autocrop_structure(image: Image.Image, pad: int = 15) -> Image.Image:
    """Tighten crop to the ink bounding box of the largest connected component.

    Designed for patent table cells where the structure occupies ~30-40% of an
    804×500+ px crop, with table borders, adjacent row bleed, and whitespace.

    Strategy:
    1. Remove table border lines (horizontal/vertical lines spanning >40%)
    2. Use a small morphological closing (5px) to bridge within-structure gaps
       without merging structures across table rows
    3. Pick the largest connected component and crop to its bounding box

    Args:
        image: PIL Image of a (possibly oversized) chemical structure crop.
        pad: Padding in pixels around the detected bounding box.

    Returns:
        Cropped PIL Image tightened to the main structure, or the original
        image if no meaningful structure region is found.
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')

    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    img_h, img_w = binary.shape

    # --- Step 1: Remove table border lines ---
    # Same logic as Pass 1 in clean_structure_image — lines spanning >40%
    clean = binary.copy()
    min_line_len_h = int(img_w * 0.4)
    min_line_len_v = int(img_h * 0.4)

    if min_line_len_h > 10:
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_line_len_h, 1))
        h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
        clean[h_lines > 0] = 0

    if min_line_len_v > 10:
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_line_len_v))
        v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
        clean[v_lines > 0] = 0

    # --- Step 2: Small closing to bridge within-structure gaps ---
    # 7px closing bridges bond gaps and atom labels without merging structures
    # across different table rows (which are typically 50-100px apart).
    # Tested: 5px fragments macrocycles, 7px cleanly separates adjacent rows.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed)
    if num_labels < 2:
        return image

    # Pick largest foreground component
    fg_areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)]
    fg_areas.sort(key=lambda x: x[1], reverse=True)
    main_label = fg_areas[0][0]

    x = stats[main_label, cv2.CC_STAT_LEFT]
    y = stats[main_label, cv2.CC_STAT_TOP]
    w = stats[main_label, cv2.CC_STAT_WIDTH]
    h = stats[main_label, cv2.CC_STAT_HEIGHT]

    # Safety: if bbox area < 5% of original, something went wrong
    if w * h < 0.05 * img_w * img_h:
        return image

    # Crop with padding, clamped to image bounds
    x0 = max(x - pad, 0)
    y0 = max(y - pad, 0)
    x1 = min(x + w + pad, img_w)
    y1 = min(y + h + pad, img_h)

    return image.crop((x0, y0, x1, y1))


def clean_structure_image(image: Image.Image, mask_text: bool = False, autocrop: bool = False) -> Image.Image:
    """Remove contaminating elements from a chemical structure image.

    Multi-pass strategy:
    Pass 0 (optional) — Use PaddleOCR to detect and white-fill non-structural
             text (compound names, captions, step labels) while preserving
             atom labels. Only runs when *mask_text* is True.
    Pass 1 — Remove long straight lines (table borders, separator lines).
             Detected via morphological line kernels. No molecular bond spans
             >50% of the image width/height, so this is safe.
    Pass 2 — Morphological closing to identify the main molecular region,
             then remove any ink outside it.

    Safe to call on clean images — returns them unchanged.

    Args:
        image: PIL Image of a cropped chemical structure.
        mask_text: If True, run PaddleOCR text masking before line removal.
        autocrop: If True, tighten the crop to the main structure bounding
            box after cleaning.

    Returns:
        Cleaned PIL Image with contaminants white-filled.
    """
    # --- Pass 0 (optional): Text masking via PaddleOCR ---
    if mask_text:
        from pdf_to_smiles.core.text_masker import mask_text_regions
        image = mask_text_regions(image)

    if image.mode != 'RGB':
        image = image.convert('RGB')

    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Binarize: ink pixels become 255 on black background
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    img_h, img_w = binary.shape
    contaminant_mask = np.zeros_like(binary)

    # --- Pass 1: Detect and remove long straight lines ---
    # Use morphological opening with long horizontal/vertical kernels to
    # isolate lines. A line must span >40% of image width/height to qualify.
    # No molecular bond would be this long relative to the image.
    min_line_len_h = int(img_w * 0.4)
    min_line_len_v = int(img_h * 0.4)

    if min_line_len_h > 10:
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_line_len_h, 1))
        h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
        # Only remove lines that are thin (height < 10% of image)
        if cv2.countNonZero(h_lines) > 0:
            nl, lb, st, _ = cv2.connectedComponentsWithStats(h_lines)
            for i in range(1, nl):
                h = st[i, cv2.CC_STAT_HEIGHT]
                if h < img_h * 0.1:
                    contaminant_mask[lb == i] = 255

    if min_line_len_v > 10:
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_line_len_v))
        v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
        if cv2.countNonZero(v_lines) > 0:
            nl, lb, st, _ = cv2.connectedComponentsWithStats(v_lines)
            for i in range(1, nl):
                w = st[i, cv2.CC_STAT_WIDTH]
                if w < img_w * 0.1:
                    contaminant_mask[lb == i] = 255

    # Remove detected lines from binary before Pass 2 so they don't
    # bridge the molecule to nearby contaminants during closing
    binary_cleaned = binary.copy()
    binary_cleaned[contaminant_mask == 255] = 0

    # --- Pass 2: Morphological closing to find main molecular region ---
    # Closing bridges small gaps in the molecule (from gray ink binarization)
    # while keeping spatially distant contaminants separate.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(binary_cleaned, cv2.MORPH_CLOSE, kernel)

    # Dilate to merge nearby closed regions
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    closed = cv2.dilate(closed, dilate_kernel)

    # Find connected components on the closed/dilated image
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed)

    if num_labels > 2:
        # The largest closed component is the main molecular region
        fg_areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)]
        fg_areas.sort(key=lambda x: x[1], reverse=True)
        main_label = fg_areas[0][0]

        # Pixels in original binary (after line removal) outside the main
        # closed region are contaminants
        main_region = (labels == main_label)
        contaminant_mask |= ((binary_cleaned == 255) & ~main_region).astype(np.uint8) * 255

    # --- Final safety check ---
    contaminant_pixels = cv2.countNonZero(contaminant_mask)
    total_ink = cv2.countNonZero(binary)
    if contaminant_pixels == 0 or contaminant_pixels > total_ink * 0.5:
        return image

    # White-fill contaminant regions
    cleaned = img_array.copy()
    cleaned[contaminant_mask == 255] = [255, 255, 255]

    result = Image.fromarray(cleaned)
    if autocrop:
        result = autocrop_structure(result)
    return result
