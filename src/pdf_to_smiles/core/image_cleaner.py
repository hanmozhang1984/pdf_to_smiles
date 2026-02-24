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


def clean_structure_image(image: Image.Image) -> Image.Image:
    """Remove contaminating elements from a chemical structure image.

    Two-pass strategy:
    Pass 1 — Remove long straight lines (table borders, separator lines).
             Detected via morphological line kernels. No molecular bond spans
             >50% of the image width/height, so this is safe.
    Pass 2 — Morphological closing to identify the main molecular region,
             then remove any ink outside it.

    Safe to call on clean images — returns them unchanged.

    Args:
        image: PIL Image of a cropped chemical structure.

    Returns:
        Cleaned PIL Image with contaminants white-filled.
    """
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

    return Image.fromarray(cleaned)
