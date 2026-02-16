"""Lightweight chemical structure detection using OpenCV contour analysis.

No ML model required — uses adaptive thresholding and contour filtering
to detect chemical structure regions on a page. Pure OpenCV (already a dependency).

Two-pass strategy:
  Pass 1: Find candidate regions using morphological close + contour detection.
  Pass 2: For large regions (likely tables containing structures), subdivide
           by finding internal contours on the original binary image.

Text vs structure discrimination:
  1. Geometry — aspect ratio, page-width span, minimum size
  2. Fill ratio — text is denser than structure line drawings
  3. Text line detection — text blocks have regular horizontal line spacing
  4. Hough lines — structures have straight bond lines
"""

from typing import List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image


class LightweightDetector:
    """Detects and segments chemical structures using OpenCV contours.

    Same interface as StructureDetector but requires no model download.
    """

    MIN_STRUCTURE_SIZE = 60   # Minimum width/height in pixels
    MAX_ASPECT_RATIO = 4.0    # Filter elongated shapes (likely text lines)
    MIN_AREA = 5000           # Minimum contour area to filter noise

    # Morphological kernel size for closing gaps in structure bonds
    MORPH_KERNEL_SIZE = 15
    # Padding around detected regions (pixels)
    CROP_PADDING = 10

    # Filters for individual structure candidates
    MAX_FILL_RATIO = 0.45     # Text blocks are very dense; structures are sparser
    MIN_FILL_RATIO = 0.005    # Too sparse = noise
    MAX_PAGE_WIDTH_RATIO = 0.50  # Skip regions spanning >50% of page width
    TEXT_FILL_THRESHOLD = 0.10   # Fill above this + text lines → likely text block

    # Text line detection thresholds
    MAX_TEXT_LINES = 3         # Reject regions with more text lines than this
    TEXT_LINE_THRESHOLD = 0.15 # Row dark pixel ratio to count as text line
    TEXT_LINE_GAP_MIN = 5      # Min gap rows between distinct text lines

    # Large region subdivision thresholds
    LARGE_REGION_AREA = 200000   # Regions larger than this get subdivided
    SUB_MIN_AREA = 5000          # Minimum area for sub-contours
    SUB_MIN_SIZE = 80            # Minimum width/height for sub-contours (filters text fragments)
    SUB_MAX_ASPECT_RATIO = 5.0   # Max aspect ratio for sub-contours
    SUB_MAX_FILL_RATIO = 0.50    # Max fill for sub-contours (slightly more permissive)
    SUB_MIN_FILL_RATIO = 0.02    # Min fill for sub-contours (reject near-empty regions)
    SUB_MAX_TEXT_LINES = 8       # Max text lines in sub-contour (atom labels count as text)
    SUB_MAX_TEXT_DENSITY = 4.0   # Max text lines per 100px height (chemical names are denser)
    SUB_MORPH_KERNEL = 8         # Smaller kernel for within-table detection

    def detect_structures(self, page_image: Image.Image) -> List[Image.Image]:
        """Detect and extract chemical structures from a page image.

        Args:
            page_image: PIL Image of a PDF page.

        Returns:
            List of PIL Images, each containing a detected chemical structure.
        """
        try:
            img_array = np.array(page_image.convert('RGB'))
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            page_height, page_width = gray.shape

            # Adaptive threshold
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, blockSize=25, C=10
            )

            # Morphological close to merge nearby strokes
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (self.MORPH_KERNEL_SIZE, self.MORPH_KERNEL_SIZE)
            )
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            # Find contours
            contours, _ = cv2.findContours(
                closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            candidates = []
            for contour in contours:
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)

                # Pass 2: Large regions (tables) — subdivide and extract
                if area > self.LARGE_REGION_AREA and w > self.MIN_STRUCTURE_SIZE * 2:
                    sub_results = self._extract_from_table_region(
                        binary, img_array, x, y, w, h, page_width, page_height
                    )
                    candidates.extend(sub_results)
                    continue

                # Pass 1: Normal-sized regions — evaluate directly
                result = self._evaluate_candidate(
                    contour, binary, img_array, page_width, page_height
                )
                if result is not None:
                    candidates.append(result)

            # Sort by position: top-to-bottom, then left-to-right
            candidates.sort(key=lambda c: (c[0], c[1]))
            return [img for _, _, img in candidates]

        except Exception as e:
            raise RuntimeError(f"Lightweight structure detection failed: {e}") from e

    def _remove_table_lines(self, binary_roi: np.ndarray) -> np.ndarray:
        """Remove long horizontal and vertical lines (table borders) from a binary ROI.

        Detects lines spanning a significant fraction of the ROI dimensions and
        erases them, leaving only cell content (structures, text, etc.).
        """
        h, w = binary_roi.shape
        cleaned = binary_roi.copy()

        # Detect horizontal lines: kernel wide enough to span table columns
        horiz_kernel_len = max(w // 4, 40)
        horiz_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (horiz_kernel_len, 1)
        )
        horiz_lines = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, horiz_kernel)

        # Detect vertical lines: kernel tall enough to span table rows
        vert_kernel_len = max(h // 4, 40)
        vert_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, vert_kernel_len)
        )
        vert_lines = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, vert_kernel)

        # Combine and dilate slightly to ensure full removal
        grid_lines = cv2.bitwise_or(horiz_lines, vert_lines)
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        grid_lines = cv2.dilate(grid_lines, dilate_kernel, iterations=1)

        # Erase the grid lines from the ROI
        cleaned = cv2.subtract(cleaned, grid_lines)
        return cleaned

    def _extract_from_table_region(
        self,
        binary: np.ndarray,
        img_array: np.ndarray,
        rx: int, ry: int, rw: int, rh: int,
        page_width: int, page_height: int,
    ) -> List[Tuple[int, int, Image.Image]]:
        """Extract individual structures from within a large region (e.g. a table).

        Removes table grid lines first, then uses a smaller morphological kernel
        to find individual structure contours within cells.
        """
        # Extract the ROI from the original binary image
        roi = binary[ry:ry+rh, rx:rx+rw]

        # Remove table grid lines so cells become separate contours
        cleaned_roi = self._remove_table_lines(roi)

        # Smaller morph kernel to keep structures separate from text
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.SUB_MORPH_KERNEL, self.SUB_MORPH_KERNEL)
        )
        closed_roi = cv2.morphologyEx(cleaned_roi, cv2.MORPH_CLOSE, kernel)

        # Find sub-contours within this region
        sub_contours, _ = cv2.findContours(
            closed_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        results = []
        for contour in sub_contours:
            area = cv2.contourArea(contour)
            if area < self.SUB_MIN_AREA:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            if w < self.SUB_MIN_SIZE or h < self.SUB_MIN_SIZE:
                continue

            aspect_ratio = max(w, h) / max(min(w, h), 1)
            if aspect_ratio > self.SUB_MAX_ASPECT_RATIO:
                continue

            # Fill ratio check — use cleaned ROI (grid lines removed)
            sub_roi = cleaned_roi[y:y+h, x:x+w]
            dark_pixels = cv2.countNonZero(sub_roi)
            fill_ratio = dark_pixels / (w * h)
            if fill_ratio > self.SUB_MAX_FILL_RATIO or fill_ratio < self.SUB_MIN_FILL_RATIO:
                continue

            # Text line check (stricter for sub-regions)
            text_lines = self._count_text_lines(sub_roi)
            if text_lines > self.SUB_MAX_TEXT_LINES:
                continue

            # Text density check — chemical names have many text lines
            # packed into a small height; structures are sparser
            text_density = text_lines / max(h, 1) * 100
            if text_density > self.SUB_MAX_TEXT_DENSITY:
                continue

            # Combined check: high fill + multiple text lines = text block
            # Use higher threshold for sub-contours (residual grid artifacts raise fill)
            if fill_ratio > 0.13 and text_lines >= 3:
                continue

            # Must have line features (bonds)
            if not self._has_line_features(sub_roi):
                continue

            # Map coordinates back to page space
            abs_x = rx + x
            abs_y = ry + y
            x1 = max(0, abs_x - self.CROP_PADDING)
            y1 = max(0, abs_y - self.CROP_PADDING)
            x2 = min(page_width, abs_x + w + self.CROP_PADDING)
            y2 = min(page_height, abs_y + h + self.CROP_PADDING)

            crop = img_array[y1:y2, x1:x2]
            results.append((y1, x1, Image.fromarray(crop)))

        return results

    def _evaluate_candidate(
        self,
        contour,
        binary: np.ndarray,
        img_array: np.ndarray,
        page_width: int,
        page_height: int,
    ) -> Optional[Tuple[int, int, Image.Image]]:
        """Evaluate a contour candidate — return (y, x, image) or None if rejected."""

        area = cv2.contourArea(contour)
        if area < self.MIN_AREA:
            return None

        x, y, w, h = cv2.boundingRect(contour)

        if w < self.MIN_STRUCTURE_SIZE or h < self.MIN_STRUCTURE_SIZE:
            return None

        aspect_ratio = max(w, h) / max(min(w, h), 1)
        if aspect_ratio > self.MAX_ASPECT_RATIO:
            return None

        if w > page_width * self.MAX_PAGE_WIDTH_RATIO:
            return None

        roi = binary[y:y+h, x:x+w]
        dark_pixels = cv2.countNonZero(roi)
        fill_ratio = dark_pixels / (w * h)

        if fill_ratio > self.MAX_FILL_RATIO or fill_ratio < self.MIN_FILL_RATIO:
            return None

        text_lines = self._count_text_lines(roi)
        if text_lines > self.MAX_TEXT_LINES:
            return None

        # Combined check: high fill + multiple text lines = text block (chemical name)
        if fill_ratio > self.TEXT_FILL_THRESHOLD and text_lines >= 3:
            return None

        if not self._has_line_features(roi):
            return None

        x1 = max(0, x - self.CROP_PADDING)
        y1 = max(0, y - self.CROP_PADDING)
        x2 = min(page_width, x + w + self.CROP_PADDING)
        y2 = min(page_height, y + h + self.CROP_PADDING)

        crop = img_array[y1:y2, x1:x2]
        return (y1, x1, Image.fromarray(crop))

    def _count_text_lines(self, binary_roi: np.ndarray) -> int:
        """Count the number of horizontal text-line bands in a region."""
        h, w = binary_roi.shape
        if h < 20 or w < 20:
            return 0

        row_sums = np.sum(binary_roi > 0, axis=1) / w
        is_text_row = row_sums > self.TEXT_LINE_THRESHOLD

        text_lines = 0
        in_line = False
        gap_count = 0

        for is_dark in is_text_row:
            if is_dark:
                if not in_line:
                    text_lines += 1
                    in_line = True
                gap_count = 0
            else:
                gap_count += 1
                if gap_count >= self.TEXT_LINE_GAP_MIN:
                    in_line = False

        return text_lines

    def _has_line_features(self, binary_roi: np.ndarray) -> bool:
        """Check if the region contains straight line segments (bonds)."""
        edges = cv2.Canny(binary_roi, 50, 150)

        h, w = binary_roi.shape
        min_line_length = max(15, min(h, w) // 8)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=20,
            minLineLength=min_line_length,
            maxLineGap=5
        )

        if lines is None:
            return False

        return len(lines) >= 3
