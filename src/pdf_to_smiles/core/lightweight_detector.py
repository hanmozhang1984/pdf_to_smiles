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
    MORPH_KERNEL_SIZE = 10
    # Padding around detected regions (pixels)
    CROP_PADDING = 15

    # Filters for individual structure candidates
    MAX_FILL_RATIO = 0.45     # Text blocks are very dense; structures are sparser
    MIN_FILL_RATIO = 0.005    # Too sparse = noise
    MAX_PAGE_WIDTH_RATIO = 0.50  # Skip regions spanning >50% of page width
    TEXT_FILL_THRESHOLD = 0.15   # Fill above this + text lines → likely text block

    # Text line detection thresholds
    MAX_TEXT_LINES = 5         # Reject regions with more text lines than this
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

    @staticmethod
    def _compute_padding(w: int, h: int) -> int:
        """Compute adaptive padding based on structure dimensions."""
        return max(15, min(50, int(0.08 * (w + h) / 2)))

    def detect_structures(self, page_image: Image.Image) -> List[Image.Image]:
        """Detect and extract chemical structures from a page image.

        Args:
            page_image: PIL Image of a PDF page.

        Returns:
            List of PIL Images, each containing a detected chemical structure.
        """
        candidates = self._detect_candidates(page_image)
        return [img for _, _, img, _ in candidates]

    def detect_structures_with_boxes(
        self, page_image: Image.Image
    ) -> List[Tuple[Image.Image, Tuple[int, int, int, int]]]:
        """Detect structures and return images with their bounding boxes.

        Args:
            page_image: PIL Image of a PDF page.

        Returns:
            List of (cropped_image, (x1, y1, x2, y2)) tuples.
        """
        candidates = self._detect_candidates(page_image)
        return [(img, box) for _, _, img, box in candidates]

    def _detect_candidates(
        self, page_image: Image.Image
    ) -> List[Tuple[int, int, Image.Image, Tuple[int, int, int, int]]]:
        """Core detection returning (y, x, image, bbox) tuples sorted by position."""
        try:
            img_array = np.array(page_image.convert('RGB'))
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            page_height, page_width = gray.shape

            # Adaptive threshold
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, blockSize=25, C=10
            )

            # Remove long horizontal lines (reaction arrows) before closing.
            # These bridge separate structures and cause them to merge into
            # one contour.  Only remove lines that are long relative to the
            # page width (>8%) — bond lines within structures are shorter.
            arrow_len = max(page_width // 12, 60)
            h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (arrow_len, 1))
            arrows = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
            binary_no_arrows = cv2.subtract(binary, arrows)

            # Morphological close to merge nearby strokes
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (self.MORPH_KERNEL_SIZE, self.MORPH_KERNEL_SIZE)
            )
            closed = cv2.morphologyEx(binary_no_arrows, cv2.MORPH_CLOSE, kernel)

            # Find contours
            contours, _ = cv2.findContours(
                closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Collect candidates with both padded and unpadded boxes
            # Internal format: (y, x, image, padded_box, unpadded_box)
            candidates_with_unpadded = []
            for contour in contours:
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)

                # Pass 2: Large regions (tables) — subdivide and extract
                if area > self.LARGE_REGION_AREA and w > self.MIN_STRUCTURE_SIZE * 2:
                    sub_results = self._extract_from_table_region(
                        binary, img_array, x, y, w, h, page_width, page_height
                    )
                    candidates_with_unpadded.extend(sub_results)
                    continue

                # Pass 1: Normal-sized regions — evaluate directly
                result = self._evaluate_candidate(
                    contour, binary, img_array, page_width, page_height
                )
                if result is not None:
                    candidates_with_unpadded.append(result)

            # Suppress overlapping padding between adjacent structures
            candidates_with_unpadded = self._suppress_overlaps(
                candidates_with_unpadded, img_array
            )

            # Strip unpadded box for public interface: (y, x, image, padded_box)
            candidates = [
                (cy, cx, img, pbox)
                for cy, cx, img, pbox, _ubox in candidates_with_unpadded
            ]

            # Sort by position: top-to-bottom, then left-to-right
            candidates.sort(key=lambda c: (c[0], c[1]))
            return candidates

        except Exception as e:
            raise RuntimeError(f"Lightweight structure detection failed: {e}") from e

    @staticmethod
    def _suppress_overlaps(candidates, img_array: np.ndarray):
        """Trim overlapping padding between adjacent structure boxes.

        When two padded bounding boxes overlap, split the shared space at the
        midpoint of the *unpadded* edges. Never trims below the original
        contour bounding box — only padding is reduced, not structure content.

        Args:
            candidates: list of (y, x, image, padded_box, unpadded_box)
            img_array: page image array for re-cropping
        """
        if len(candidates) <= 1:
            return candidates

        padded = [list(c[3]) for c in candidates]
        unpadded = [c[4] for c in candidates]

        for i in range(len(padded)):
            for j in range(i + 1, len(padded)):
                ax1, ay1, ax2, ay2 = padded[i]
                bx1, by1, bx2, by2 = padded[j]

                # Check if padded boxes overlap at all
                if ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1:
                    continue

                uax1, uay1, uax2, uay2 = unpadded[i]
                ubx1, uby1, ubx2, uby2 = unpadded[j]

                # Vertical overlap: A is above B
                if ay2 > by1 and ay1 < by1:
                    # Split at midpoint of unpadded edges
                    mid_y = (uay2 + uby1) // 2
                    # Trim but never below unpadded box
                    padded[i][3] = max(uay2, min(ay2, mid_y))
                    padded[j][1] = min(uby1, max(by1, mid_y))

                # Horizontal overlap: A is left of B
                if ax2 > bx1 and ax1 < bx1:
                    mid_x = (uax2 + ubx1) // 2
                    padded[i][2] = max(uax2, min(ax2, mid_x))
                    padded[j][0] = min(ubx1, max(bx1, mid_x))

        # Re-crop images using adjusted boxes
        result = []
        for idx, cand in enumerate(candidates):
            x1, y1, x2, y2 = padded[idx]
            if x2 <= x1 or y2 <= y1:
                continue
            crop = img_array[y1:y2, x1:x2]
            box = (x1, y1, x2, y2)
            result.append((y1, x1, Image.fromarray(crop), box, cand[4]))
        return result

    def _remove_table_lines(
        self, binary_roi: np.ndarray
    ) -> Tuple[np.ndarray, List[int], List[int]]:
        """Remove long horizontal and vertical lines (table borders) from a binary ROI.

        Detects lines spanning a significant fraction of the ROI dimensions and
        erases them, leaving only cell content (structures, text, etc.).

        Returns:
            (cleaned_roi, h_positions, v_positions) where h_positions and
            v_positions are sorted lists of Y/X coordinates of detected grid lines.
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

        # Extract line positions from the detected line masks
        h_positions = self._extract_line_positions(horiz_lines, axis=1)
        v_positions = self._extract_line_positions(vert_lines, axis=0)

        # Combine and dilate slightly to ensure full removal
        grid_lines = cv2.bitwise_or(horiz_lines, vert_lines)
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        grid_lines = cv2.dilate(grid_lines, dilate_kernel, iterations=1)

        # Erase the grid lines from the ROI
        cleaned = cv2.subtract(cleaned, grid_lines)
        return cleaned, h_positions, v_positions

    @staticmethod
    def _extract_line_positions(line_mask: np.ndarray, axis: int) -> List[int]:
        """Extract sorted positions of detected lines along the given axis.

        Args:
            line_mask: Binary mask of detected lines.
            axis: 1 for horizontal lines (returns Y positions),
                  0 for vertical lines (returns X positions).

        Returns:
            Sorted list of line center positions.
        """
        # Project along the perpendicular axis
        projection = np.sum(line_mask > 0, axis=axis)
        threshold = max(1, line_mask.shape[axis] // 8)

        positions = []
        in_line = False
        start = 0

        for i, val in enumerate(projection):
            if val >= threshold:
                if not in_line:
                    start = i
                    in_line = True
            else:
                if in_line:
                    positions.append((start + i) // 2)  # center of line
                    in_line = False

        if in_line:
            positions.append((start + len(projection)) // 2)

        return sorted(positions)

    def _extract_from_table_region(
        self,
        binary: np.ndarray,
        img_array: np.ndarray,
        rx: int, ry: int, rw: int, rh: int,
        page_width: int, page_height: int,
    ) -> List[Tuple[int, int, Image.Image, Tuple[int, int, int, int]]]:
        """Extract individual structures from within a large region (e.g. a table).

        Removes table grid lines first, then uses a smaller morphological kernel
        to find individual structure contours within cells.
        """
        # Extract the ROI from the original binary image
        roi = binary[ry:ry+rh, rx:rx+rw]

        # Remove table grid lines so cells become separate contours
        cleaned_roi, h_positions, v_positions = self._remove_table_lines(roi)

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

            # Must have line features (bonds) — relaxed threshold for table
            # sub-contours since column context already confirms structure
            if not self._has_line_features(sub_roi, min_lines=1):
                continue

            # Map coordinates back to page space
            abs_x = rx + x
            abs_y = ry + y
            unpadded_box = (abs_x, abs_y, abs_x + w, abs_y + h)
            padding = self._compute_padding(w, h)

            # Find enclosing cell boundaries from grid lines
            cell_x1 = max((vp for vp in v_positions if vp <= x), default=0)
            cell_x2 = min((vp for vp in v_positions if vp >= x + w), default=rw)
            cell_y1 = max((hp for hp in h_positions if hp <= y), default=0)
            cell_y2 = min((hp for hp in h_positions if hp >= y + h), default=rh)

            # Clamp padding to stay within the cell (positions are ROI-relative)
            # But never trim tighter than the unpadded contour box
            x1 = max(rx + cell_x1, abs_x - padding)
            y1 = max(ry + cell_y1, abs_y - padding)
            x2 = min(rx + cell_x2, abs_x + w + padding)
            y2 = min(ry + cell_y2, abs_y + h + padding)

            # Ensure we never crop into the actual structure content
            x1 = min(x1, abs_x)
            y1 = min(y1, abs_y)
            x2 = max(x2, abs_x + w)
            y2 = max(y2, abs_y + h)

            # Clamp to page boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(page_width, x2)
            y2 = min(page_height, y2)

            crop = img_array[y1:y2, x1:x2]
            results.append((y1, x1, Image.fromarray(crop), (x1, y1, x2, y2), unpadded_box))

        return results

    def _evaluate_candidate(
        self,
        contour,
        binary: np.ndarray,
        img_array: np.ndarray,
        page_width: int,
        page_height: int,
    ) -> Optional[tuple]:
        """Evaluate a contour candidate.

        Returns (y, x, image, padded_box, unpadded_box) or None if rejected.
        """
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

        unpadded_box = (x, y, x + w, y + h)

        padding = self._compute_padding(w, h)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(page_width, x + w + padding)
        y2 = min(page_height, y + h + padding)

        crop = img_array[y1:y2, x1:x2]
        return (y1, x1, Image.fromarray(crop), (x1, y1, x2, y2), unpadded_box)

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

    def _has_line_features(self, binary_roi: np.ndarray, min_lines: int = 3) -> bool:
        """Check if the region contains straight line segments (bonds).

        Args:
            binary_roi: Binary image of the region.
            min_lines: Minimum number of Hough line segments required.
                Default 3 for standalone candidates; use 1 for table
                sub-contours where column context already confirms structure.
        """
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

        return len(lines) >= min_lines
