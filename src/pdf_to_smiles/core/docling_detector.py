"""Structure detector using cached Docling layout analysis.

Wraps DoclingClassifier to provide structure detection with bounding boxes,
reusing the RT-DETR layout clusters that are already computed during page
scanning. This avoids the heuristic OpenCV pipeline (LightweightDetector)
which can truncate/miscrop structures on complex synthesis pages.
"""

import logging
import re
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class DoclingDetector:
    """Detect chemical structures using cached Docling layout bounding boxes.

    Requires a DoclingClassifier that has already run classify_pdf() or
    detect_structure_pages() so that layout clusters are cached.

    Unlike LightweightDetector (which is stateless), this detector needs
    page_num to look up cached layout data.
    """

    STRUCTURE_LABELS = {"picture", "formula", "chart"}
    PADDING_PX = 15

    # Thresholds for _merge_split_boxes
    _MERGE_Y_OVERLAP_RATIO = 0.3
    _MERGE_X_GAP_PX = 50

    # Thresholds for ink detection
    _INK_DARK_THRESHOLD = 180  # grayscale value below which a pixel is "dark"
    _INK_RATIO_THRESHOLD = 0.01  # fraction of dark pixels to count as ink
    _STRUCTURE_INK_MIN_CELLS = 6  # min occupied cells in 4x4 grid
    _STRUCTURE_INK_CELL_THRESHOLD = 0.005  # min dark ratio per cell

    # Minimum row height for a candidate structure cell (px at 200 DPI).
    # Header rows are typically 40-110 px; real structure cells are 200+.
    _MIN_STRUCTURE_ROW_HEIGHT = 160

    # Template matching — threshold is intentionally low because different
    # chemical structures have near-zero cross-correlation; we only want
    # to reject truly anti-correlated (inverted/blank) regions.
    _TEMPLATE_SIMILARITY_THRESHOLD = -0.3

    # Keyword for table caption/header scanning — only "structure" is safe;
    # "example" and "table \d" also match text-only tables (TABLE 2, TABLE 3).
    _TABLE_CAPTION_KEYWORDS = re.compile(r"structure", re.IGNORECASE)

    def __init__(self, docling_classifier):
        """Initialize with a DoclingClassifier that has cached layout data.

        Args:
            docling_classifier: A DoclingClassifier instance that has already
                processed the PDF (layout clusters are cached).
        """
        self._classifier = docling_classifier
        self._table_row_cache = None  # (x_left, x_right) — structure column X-range
        self._template_crop = None   # grayscale np.array of a known structure

    def detect_structures_with_boxes(
        self,
        page_image: Image.Image,
        page_num: int,
    ) -> List[Tuple[Image.Image, Tuple[int, int, int, int]]]:
        """Detect structures on a page using cached Docling layout data.

        Args:
            page_image: PIL Image of the rendered PDF page.
            page_num: 1-indexed page number (for looking up cached layout).

        Returns:
            List of (cropped_image, (x1, y1, x2, y2)) tuples.
            Empty list if no structures found on this page.
        """
        boxes = self._classifier.get_structure_boxes(page_num, page_image.size)
        boxes = self._merge_split_boxes(boxes)
        boxes = self._fill_table_gaps(boxes, page_image, page_num)
        boxes = self._scan_example_tables(boxes, page_image, page_num)
        boxes = self._close_row_gaps(boxes, page_num, page_image.size)
        boxes = self._split_compound_boxes(boxes, page_image, page_num)
        if not boxes:
            return []

        img_w, img_h = page_image.size
        results = []

        for (x1, y1, x2, y2) in boxes:
            # Add padding, clamped to image bounds
            px1 = max(0, x1 - self.PADDING_PX)
            py1 = max(0, y1 - self.PADDING_PX)
            px2 = min(img_w, x2 + self.PADDING_PX)
            py2 = min(img_h, y2 + self.PADDING_PX)

            cropped = page_image.crop((px1, py1, px2, py2))
            results.append((cropped, (px1, py1, px2, py2)))

        logger.debug(
            "DoclingDetector: page %d found %d structure regions",
            page_num, len(results),
        )
        return results

    # ------------------------------------------------------------------
    # Post-processing: merge horizontally-split boxes
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_split_boxes(
        boxes: List[Tuple[int, int, int, int]],
    ) -> List[Tuple[int, int, int, int]]:
        """Merge boxes that were split horizontally at table column boundaries.

        Two boxes are candidates for merging when they overlap vertically
        (>=30% of the shorter box's height) and are horizontally adjacent
        or overlapping (gap <=50 px).  Transitive merges are handled via
        union-find so that A-B and B-C all collapse into one box.

        Returns a new list of (x1, y1, x2, y2) bounding boxes.
        """
        if len(boxes) <= 1:
            return list(boxes)

        n = len(boxes)
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i, j):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj

        for i in range(n):
            ax1, ay1, ax2, ay2 = boxes[i]
            ah = ay2 - ay1
            for j in range(i + 1, n):
                bx1, by1, bx2, by2 = boxes[j]
                bh = by2 - by1

                # Y-overlap
                overlap_y = max(0, min(ay2, by2) - max(ay1, by1))
                if overlap_y / max(min(ah, bh), 1) < DoclingDetector._MERGE_Y_OVERLAP_RATIO:
                    continue

                # X-gap (positive = gap, negative = overlap)
                x_gap = max(0, max(ax1, bx1) - min(ax2, bx2))
                if x_gap > DoclingDetector._MERGE_X_GAP_PX:
                    continue

                union(i, j)

        # Collect groups and compute bounding unions
        from collections import defaultdict
        groups = defaultdict(list)
        for i in range(n):
            groups[find(i)].append(i)

        merged = []
        for idxs in groups.values():
            xs1 = min(boxes[i][0] for i in idxs)
            ys1 = min(boxes[i][1] for i in idxs)
            xs2 = max(boxes[i][2] for i in idxs)
            ys2 = max(boxes[i][3] for i in idxs)
            merged.append((xs1, ys1, xs2, ys2))

        if len(merged) < n:
            logger.debug(
                "_merge_split_boxes: merged %d boxes into %d", n, len(merged),
            )

        return merged

    # ------------------------------------------------------------------
    # Post-processing: fill missed table rows
    # ------------------------------------------------------------------

    def _get_table_boxes(
        self, page_num: int, page_image_size: Tuple[int, int],
    ) -> List[Tuple[int, int, int, int]]:
        """Return pixel-coordinate bounding boxes for table regions on a page."""
        clusters = self._classifier._cached_layout.get(page_num, [])
        if not clusters:
            return []

        img_w, img_h = page_image_size
        page_dims = self._classifier._cached_page_dims.get(page_num)

        table_boxes = []
        for label, conf, bbox in clusters:
            if label != "table":
                continue

            if page_dims is not None:
                page_w_pts, page_h_pts = page_dims
                try:
                    from docling_core.types.doc.base import CoordOrigin
                    if bbox.coord_origin == CoordOrigin.BOTTOMLEFT:
                        tl_bbox = bbox.to_top_left_origin(page_height=page_h_pts)
                    else:
                        tl_bbox = bbox

                    scale_x = img_w / page_w_pts
                    scale_y = img_h / page_h_pts
                    x1 = int(tl_bbox.l * scale_x)
                    y1 = int(tl_bbox.t * scale_y)
                    x2 = int(tl_bbox.r * scale_x)
                    y2 = int(tl_bbox.b * scale_y)
                except Exception:
                    scale_x = img_w / page_w_pts
                    scale_y = img_h / page_h_pts
                    x1 = int(bbox.l * scale_x)
                    y1 = int(bbox.t * scale_y)
                    x2 = int(bbox.r * scale_x)
                    y2 = int(bbox.b * scale_y)
            else:
                x1, y1, x2, y2 = int(bbox.l), int(bbox.t), int(bbox.r), int(bbox.b)

            if y1 > y2:
                y1, y2 = y2, y1
            if x1 > x2:
                x1, x2 = x2, x1

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_w, x2)
            y2 = min(img_h, y2)

            if x2 - x1 >= 10 and y2 - y1 >= 10:
                table_boxes.append((x1, y1, x2, y2))

        return table_boxes

    # ------------------------------------------------------------------
    # Line-based table boundary detection
    # ------------------------------------------------------------------

    _BOUNDARY_MERGE_PX = 15  # merge detected lines within this distance

    @staticmethod
    def _merge_close_boundaries(positions: List[int]) -> List[int]:
        """Merge boundary positions that are within _BOUNDARY_MERGE_PX of each other.

        When table border lines are detected separately from table edge
        coordinates, they create tiny 2-3 px "segments".  This merges them
        by averaging clusters of positions that are close together.
        """
        if len(positions) <= 1:
            return list(positions)

        merged: List[int] = []
        cluster = [positions[0]]
        for p in positions[1:]:
            if p - cluster[-1] <= DoclingDetector._BOUNDARY_MERGE_PX:
                cluster.append(p)
            else:
                merged.append(sum(cluster) // len(cluster))
                cluster = [p]
        merged.append(sum(cluster) // len(cluster))
        return merged

    def _detect_table_row_boundaries(
        self, page_image: Image.Image,
        tx1: int, ty1: int, tx2: int, ty2: int,
    ) -> List[int]:
        """Detect horizontal separator lines in a table region.

        Returns sorted list of absolute Y-coordinates of row boundaries
        (including table top and bottom edges).
        """
        gray = np.array(page_image.crop((tx1, ty1, tx2, ty2)).convert("L"))
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        h, w = binary.shape
        kernel_len = max(w // 3, 80)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
        h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

        projection = np.sum(h_lines > 0, axis=1)
        threshold = max(1, w // 8)

        positions: List[int] = []
        in_line = False
        start = 0
        for i, val in enumerate(projection):
            if val >= threshold:
                if not in_line:
                    start = i
                    in_line = True
            else:
                if in_line:
                    positions.append((start + i) // 2)
                    in_line = False
        if in_line:
            positions.append((start + h) // 2)

        abs_positions = sorted(set([ty1] + [ty1 + p for p in positions] + [ty2]))
        merged = self._merge_close_boundaries(abs_positions)

        # If any segment is too tall, subdivide using whitespace gaps.
        # This handles tables without horizontal separator lines (e.g. GLP).
        merged = self._subdivide_tall_segments(
            page_image, tx1, tx2, merged,
        )

        return merged

    # Maximum height (px) of a single row segment before we attempt
    # whitespace-based subdivision.  At 200 DPI a single structure row
    # is typically 300-450 px; 600 px safely avoids false splits.
    _MAX_ROW_SEGMENT_HEIGHT = 600

    # Minimum whitespace gap height (px) required to split a segment.
    # Must be wide enough to avoid splitting within a row's internal
    # spacing but narrow enough to catch actual inter-row gaps.
    _MIN_WHITESPACE_GAP = 30

    # Narrower gaps are accepted if the minimum density within the gap
    # is effectively zero (< 0.5%).  This catches tables where inter-row
    # whitespace is narrow but completely blank.
    _NARROW_GAP_MIN = 15
    _NARROW_GAP_DENSITY = 0.005

    def _subdivide_tall_segments(
        self,
        page_image: Image.Image,
        x1: int, x2: int,
        row_bounds: List[int],
    ) -> List[int]:
        """Split row segments that are too tall using whitespace gap detection.

        For each segment taller than _MAX_ROW_SEGMENT_HEIGHT, analyse the
        horizontal ink-density profile and insert boundaries at large
        whitespace gaps.  Returns the refined list of boundaries.
        """
        refined: List[int] = [row_bounds[0]]

        for r in range(len(row_bounds) - 1):
            seg_y1 = row_bounds[r]
            seg_y2 = row_bounds[r + 1]
            seg_h = seg_y2 - seg_y1

            if seg_h <= self._MAX_ROW_SEGMENT_HEIGHT:
                refined.append(seg_y2)
                continue

            # Compute per-row dark-pixel density inside segment
            gray = np.array(
                page_image.crop((x1, seg_y1, x2, seg_y2)).convert("L")
            )
            dark = gray < self._INK_DARK_THRESHOLD
            row_density = np.mean(dark, axis=1).astype(np.float32)

            # Smooth to avoid splitting on single blank scanlines
            kernel_size = 15
            kernel = np.ones(kernel_size) / kernel_size
            smoothed = np.convolve(row_density, kernel, mode="same")

            # Detect contiguous whitespace runs (density < 2%)
            gap_thresh = 0.02
            in_gap = False
            gap_start = 0
            gaps: List[Tuple[int, int]] = []  # (center_y_rel, width)
            for i, val in enumerate(smoothed):
                if val < gap_thresh:
                    if not in_gap:
                        gap_start = i
                        in_gap = True
                else:
                    if in_gap:
                        gap_w = i - gap_start
                        min_density = float(smoothed[gap_start:i].min())
                        # Accept wide gaps normally, or narrow gaps if
                        # the density drops to near zero
                        if (gap_w >= self._MIN_WHITESPACE_GAP
                                or (gap_w >= self._NARROW_GAP_MIN
                                    and min_density < self._NARROW_GAP_DENSITY)):
                            gaps.append(((gap_start + i) // 2, gap_w))
                        in_gap = False
            if in_gap:
                gap_w = len(smoothed) - gap_start
                if gap_w >= self._MIN_WHITESPACE_GAP:
                    gaps.append(((gap_start + len(smoothed)) // 2, gap_w))

            if gaps:
                for center, _ in gaps:
                    refined.append(seg_y1 + center)
                logger.debug(
                    "_subdivide_tall_segments: split segment y=%d-%d (h=%d) "
                    "into %d sub-segments via %d whitespace gaps",
                    seg_y1, seg_y2, seg_h, len(gaps) + 1, len(gaps),
                )

            refined.append(seg_y2)

        return sorted(set(refined))

    def _detect_table_column_boundaries(
        self, page_image: Image.Image,
        tx1: int, ty1: int, tx2: int, ty2: int,
    ) -> List[int]:
        """Detect vertical separator lines in a table region.

        Returns sorted list of absolute X-coordinates of column boundaries.
        """
        gray = np.array(page_image.crop((tx1, ty1, tx2, ty2)).convert("L"))
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        h, w = binary.shape
        kernel_len = max(h // 3, 80)
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
        v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

        projection = np.sum(v_lines > 0, axis=0)
        threshold = max(1, h // 8)

        positions: List[int] = []
        in_line = False
        start = 0
        for i, val in enumerate(projection):
            if val >= threshold:
                if not in_line:
                    start = i
                    in_line = True
            else:
                if in_line:
                    positions.append((start + i) // 2)
                    in_line = False
        if in_line:
            positions.append((start + w) // 2)

        abs_positions = sorted(set([tx1] + [tx1 + p for p in positions] + [tx2]))
        return self._merge_close_boundaries(abs_positions)

    def _identify_structure_column(
        self, col_boundaries: List[int],
        boxes: List[Tuple], inside_idxs: List[int],
    ) -> Tuple[int, int]:
        """Return (x_left, x_right) of the structure column."""
        if inside_idxs:
            mid_xs = [(boxes[i][0] + boxes[i][2]) / 2 for i in inside_idxs]
            avg_mid = sum(mid_xs) / len(mid_xs)
            for k in range(len(col_boundaries) - 1):
                if col_boundaries[k] <= avg_mid <= col_boundaries[k + 1]:
                    return col_boundaries[k], col_boundaries[k + 1]

        # Fallback: use cached column range
        if self._table_row_cache is not None:
            cached_x_left, cached_x_right = self._table_row_cache
            cached_mid = (cached_x_left + cached_x_right) / 2
            for k in range(len(col_boundaries) - 1):
                if col_boundaries[k] <= cached_mid <= col_boundaries[k + 1]:
                    return col_boundaries[k], col_boundaries[k + 1]

        # Fallback: second column (patent tables are always
        # Example | Structure | Name | ...).  Border-artifact columns
        # have been merged away by _merge_close_boundaries so index 1
        # reliably points to the Structure column.
        if len(col_boundaries) >= 3:
            return col_boundaries[1], col_boundaries[2]

        # Last resort: full table width
        return col_boundaries[0], col_boundaries[-1]

    def _fill_table_gaps(
        self,
        boxes: List[Tuple[int, int, int, int]],
        page_image: Image.Image,
        page_num: int,
    ) -> List[Tuple[int, int, int, int]]:
        """Fill missed structure rows inside Docling-detected tables.

        Uses horizontal/vertical line detection to find actual cell boundaries
        rather than walking a rigid averaged-pitch grid.  Falls back to the
        old pitch-based approach when fewer than 3 horizontal lines are found.

        Returns the original boxes plus any validated synthetic boxes.
        """
        table_boxes = self._get_table_boxes(page_num, page_image.size)
        if not table_boxes:
            return boxes

        new_boxes = list(boxes)

        for tx1, ty1, tx2, ty2 in table_boxes:
            # Find structure boxes whose center falls inside this table
            inside_idxs = []
            for i, (bx1, by1, bx2, by2) in enumerate(boxes):
                cx = (bx1 + bx2) / 2
                cy = (by1 + by2) / 2
                if tx1 <= cx <= tx2 and ty1 <= cy <= ty2:
                    inside_idxs.append(i)

            if not inside_idxs:
                continue

            # Detect row/column boundaries from separator lines
            row_bounds = self._detect_table_row_boundaries(
                page_image, tx1, ty1, tx2, ty2,
            )
            col_bounds = self._detect_table_column_boundaries(
                page_image, tx1, ty1, tx2, ty2,
            )

            # Fallback: if line detection found too few rows, use pitch-based
            if len(row_bounds) < 3:
                row_bounds = self._pitch_fallback_rows(
                    boxes, inside_idxs, ty1, ty2,
                )

            # Identify structure column
            x_left, x_right = self._identify_structure_column(
                col_bounds, boxes, inside_idxs,
            )

            # Cache structure column X-range for cross-page rescue
            self._table_row_cache = (x_left, x_right)
            self._template_crop = np.array(
                page_image.crop(boxes[inside_idxs[0]]).convert("L")
            )

            # Skip the header row (first segment) — data rows start from
            # the second row segment onward
            data_start = 1 if len(row_bounds) > 2 else 0

            for r in range(data_start, len(row_bounds) - 1):
                cand_y1 = row_bounds[r]
                cand_y2 = row_bounds[r + 1]
                cand_x1 = max(0, x_left)
                cand_x2 = min(page_image.size[0], x_right)

                if cand_x2 - cand_x1 < 10 or cand_y2 - cand_y1 < self._MIN_STRUCTURE_ROW_HEIGHT:
                    continue

                # Check if any existing box already covers this row
                row_cy = (cand_y1 + cand_y2) / 2
                x_mid = (cand_x1 + cand_x2) / 2
                covered = False
                for bx1, by1, bx2, by2 in new_boxes:
                    if by1 <= row_cy <= by2 and bx1 <= x_mid <= bx2:
                        covered = True
                        break

                if not covered:
                    if self._validate_candidate(
                        page_image, cand_x1, cand_y1, cand_x2, cand_y2,
                    ):
                        new_boxes.append((cand_x1, cand_y1, cand_x2, cand_y2))
                        logger.debug(
                            "_fill_table_gaps: added synthetic box (%d,%d,%d,%d) on page %d",
                            cand_x1, cand_y1, cand_x2, cand_y2, page_num,
                        )

        if len(new_boxes) > len(boxes):
            logger.debug(
                "_fill_table_gaps: page %d — %d original + %d synthetic = %d total",
                page_num, len(boxes), len(new_boxes) - len(boxes), len(new_boxes),
            )

        return new_boxes

    @staticmethod
    def _pitch_fallback_rows(
        boxes: List[Tuple[int, int, int, int]],
        inside_idxs: List[int],
        ty1: int, ty2: int,
    ) -> List[int]:
        """Compute row boundaries using averaged pitch when line detection fails."""
        heights = [boxes[i][3] - boxes[i][1] for i in inside_idxs]
        avg_h = sum(heights) / len(heights)
        if avg_h < 10:
            return [ty1, ty2]

        sorted_y1s = sorted(boxes[i][1] for i in inside_idxs)

        if len(sorted_y1s) >= 2:
            intervals = [sorted_y1s[j + 1] - sorted_y1s[j]
                         for j in range(len(sorted_y1s) - 1)]
            pitch = sum(intervals) / len(intervals)
        else:
            pitch = avg_h

        # Walk up from first detection to find first row start
        first_row_y = sorted_y1s[0]
        while first_row_y - pitch >= ty1:
            first_row_y -= pitch

        # Build row boundaries
        rows = [int(first_row_y)]
        y = first_row_y + pitch
        while y < ty2 + pitch * 0.3:
            rows.append(int(min(y, ty2)))
            y += pitch

        if rows[-1] < ty2:
            rows.append(ty2)

        return sorted(set([ty1] + rows))

    # ------------------------------------------------------------------
    # Post-processing: rescue tables with 0 detections
    # ------------------------------------------------------------------

    def _scan_example_tables(
        self,
        boxes: List[Tuple[int, int, int, int]],
        page_image: Image.Image,
        page_num: int,
    ) -> List[Tuple[int, int, int, int]]:
        """Rescue structure detection for tables with 0 detected structures.

        For each table region that has no structure boxes inside it:
        1. OCR a strip above the table to check for relevant keywords
        2. Detect row/column boundaries via separator lines
        3. Identify structure column (cached X-range or pick 2nd column)
        4. Validate each candidate with _has_structure_ink + template matching

        Returns original boxes plus any validated synthetic boxes.
        """
        table_boxes = self._get_table_boxes(page_num, page_image.size)
        if not table_boxes:
            return boxes

        new_boxes = list(boxes)

        for tx1, ty1, tx2, ty2 in table_boxes:
            # Check if any existing box center falls inside this table
            has_structures = False
            for bx1, by1, bx2, by2 in boxes:
                cx = (bx1 + bx2) / 2
                cy = (by1 + by2) / 2
                if tx1 <= cx <= tx2 and ty1 <= cy <= ty2:
                    has_structures = True
                    break

            if has_structures:
                continue

            # OCR the caption above the table plus the table header/subtitle
            # to check for "Structure" keyword.
            caption_y1 = max(0, ty1 - 80)
            caption_y2 = min(page_image.size[1], ty1 + 120)
            if caption_y2 - caption_y1 < 5:
                continue

            caption_strip = page_image.crop((tx1, caption_y1, tx2, caption_y2))
            try:
                import pytesseract
                from pdf_to_smiles.utils.paths import configure_tesseract
                configure_tesseract()
                caption_text = pytesseract.image_to_string(caption_strip)
            except Exception:
                logger.debug(
                    "_scan_example_tables: OCR failed for table caption on page %d",
                    page_num,
                )
                continue

            if not self._TABLE_CAPTION_KEYWORDS.search(caption_text):
                logger.debug(
                    "_scan_example_tables: page %d table caption has no keywords: %r",
                    page_num, caption_text.strip(),
                )
                continue

            logger.debug(
                "_scan_example_tables: page %d matched caption: %r",
                page_num, caption_text.strip(),
            )

            # Detect row/column boundaries from separator lines
            row_bounds = self._detect_table_row_boundaries(
                page_image, tx1, ty1, tx2, ty2,
            )
            col_bounds = self._detect_table_column_boundaries(
                page_image, tx1, ty1, tx2, ty2,
            )

            # Fallback: if line detection found too few rows, use pitch estimate
            if len(row_bounds) < 3:
                table_h = ty2 - ty1
                if self._table_row_cache is not None:
                    # Use cached structure column width to estimate pitch
                    cached_x_left, cached_x_right = self._table_row_cache
                    header_offset = int(table_h * 0.07)
                    usable_h = table_h - header_offset
                    est_rows = max(2, min(6, round(usable_h / 400)))
                    pitch = usable_h / est_rows
                else:
                    header_offset = int(table_h * 0.07)
                    usable_h = table_h - header_offset
                    est_rows = max(2, min(6, round(usable_h / 400)))
                    pitch = usable_h / est_rows

                row_bounds = [ty1 + header_offset]
                y = ty1 + header_offset + pitch
                while y < ty2 + pitch * 0.3:
                    row_bounds.append(int(min(y, ty2)))
                    y += pitch
                if row_bounds[-1] < ty2:
                    row_bounds.append(ty2)
                row_bounds = sorted(set([ty1] + row_bounds))

            # Identify structure column
            x_left, x_right = self._identify_structure_column(
                col_bounds, boxes, [],
            )

            # If no cache yet, use full table width with margins as fallback
            if self._table_row_cache is None and len(col_bounds) < 3:
                margin = int((tx2 - tx1) * 0.05)
                x_left = tx1 + margin
                x_right = tx2 - margin

            # Cache structure column X-range for subsequent pages
            self._table_row_cache = (x_left, x_right)

            # Skip the header row
            data_start = 1 if len(row_bounds) > 2 else 0

            for r in range(data_start, len(row_bounds) - 1):
                cand_y1 = row_bounds[r]
                cand_y2 = row_bounds[r + 1]
                cand_x1 = max(0, x_left)
                cand_x2 = min(page_image.size[0], x_right)

                if cand_x2 - cand_x1 < 10 or cand_y2 - cand_y1 < self._MIN_STRUCTURE_ROW_HEIGHT:
                    continue

                # Check not already covered
                row_cy = (cand_y1 + cand_y2) / 2
                x_mid = (cand_x1 + cand_x2) / 2
                covered = False
                for bx1, by1, bx2, by2 in new_boxes:
                    if by1 <= row_cy <= by2 and bx1 <= x_mid <= bx2:
                        covered = True
                        break

                if not covered and self._validate_candidate(
                    page_image, cand_x1, cand_y1, cand_x2, cand_y2,
                ):
                    new_boxes.append((cand_x1, cand_y1, cand_x2, cand_y2))
                    logger.debug(
                        "_scan_example_tables: added synthetic box (%d,%d,%d,%d) on page %d",
                        cand_x1, cand_y1, cand_x2, cand_y2, page_num,
                    )

        if len(new_boxes) > len(boxes):
            logger.debug(
                "_scan_example_tables: page %d — %d original + %d synthetic = %d total",
                page_num, len(boxes), len(new_boxes) - len(boxes), len(new_boxes),
            )

        return new_boxes

    # ------------------------------------------------------------------
    # Post-processing: close inter-row gaps to include example numbers
    # ------------------------------------------------------------------

    def _close_row_gaps(
        self,
        boxes: List[Tuple[int, int, int, int]],
        page_num: int,
        page_image_size: Tuple[int, int],
    ) -> List[Tuple[int, int, int, int]]:
        """Expand box tops upward to close small gaps between consecutive rows.

        Inside table regions, example numbers (e.g. "74") sit in the gap
        between the previous row's bottom and the current row's top.  This
        pass extends each box's y1 upward to the previous box's y2 so that
        the example number line is included in the crop.

        Returns a new list of boxes (original boxes outside tables are unchanged).
        """
        table_boxes = self._get_table_boxes(page_num, page_image_size)
        if not table_boxes or not boxes:
            return boxes

        result = list(boxes)

        for tx1, ty1, tx2, ty2 in table_boxes:
            # Collect indices of boxes whose center falls inside this table
            inside = []
            for i, (bx1, by1, bx2, by2) in enumerate(result):
                cx = (bx1 + bx2) / 2
                cy = (by1 + by2) / 2
                if tx1 <= cx <= tx2 and ty1 <= cy <= ty2:
                    inside.append(i)

            if len(inside) < 2:
                continue

            # Sort by y1
            inside.sort(key=lambda i: result[i][1])

            # For each consecutive pair, close the gap
            for k in range(1, len(inside)):
                prev_idx = inside[k - 1]
                curr_idx = inside[k]
                prev_y2 = result[prev_idx][3]
                curr_x1, curr_y1, curr_x2, curr_y2 = result[curr_idx]
                gap = curr_y1 - prev_y2

                if 0 < gap < 80:  # small gap likely containing example number
                    result[curr_idx] = (curr_x1, prev_y2, curr_x2, curr_y2)

        return result

    # ------------------------------------------------------------------
    # Post-processing: split compound boxes (tall or wide)
    # ------------------------------------------------------------------

    # Wide boxes outside tables with aspect ratio > this trigger vertical split
    _WIDE_BOX_ASPECT_RATIO = 2.5
    # After splitting a wide box, reject if either sub-box exceeds this ratio
    _WIDE_SPLIT_MAX_ASPECT = 3.0
    # Minimum height for wide-box splitting candidates
    _WIDE_BOX_MIN_HEIGHT = 150

    def _split_compound_boxes(
        self,
        boxes: List[Tuple[int, int, int, int]],
        page_image: Image.Image,
        page_num: int,
    ) -> List[Tuple[int, int, int, int]]:
        """Split boxes that contain two structures (diastereomer pairs or side-by-side).

        Two cases:
        1. Tall boxes (height > _MAX_ROW_SEGMENT_HEIGHT): split horizontally
           at the deepest whitespace gap in the middle 60%.
        2. Wide boxes outside tables (aspect ratio > 2.5): split vertically
           at the deepest whitespace gap in the middle 60%.

        Both sub-boxes must pass validation; otherwise the original is kept.
        """
        table_boxes = self._get_table_boxes(page_num, page_image.size)
        result: List[Tuple[int, int, int, int]] = []

        for box in boxes:
            x1, y1, x2, y2 = box
            bw = x2 - x1
            bh = y2 - y1

            # --- Case 1: Tall box → horizontal split ---
            if bh > self._MAX_ROW_SEGMENT_HEIGHT:
                split = self._try_horizontal_split(page_image, x1, y1, x2, y2)
                if split is not None:
                    result.extend(split)
                    logger.debug(
                        "_split_compound_boxes: page %d tall box (%d,%d,%d,%d) h=%d → split into 2",
                        page_num, x1, y1, x2, y2, bh,
                    )
                    continue

            # --- Case 2: Wide box outside tables → vertical split ---
            if (bh >= self._WIDE_BOX_MIN_HEIGHT
                    and bw / max(bh, 1) > self._WIDE_BOX_ASPECT_RATIO
                    and not self._box_inside_any_table(box, table_boxes)):
                split = self._try_vertical_split(page_image, x1, y1, x2, y2)
                if split is not None:
                    result.extend(split)
                    logger.debug(
                        "_split_compound_boxes: page %d wide box (%d,%d,%d,%d) ratio=%.1f → split into 2",
                        page_num, x1, y1, x2, y2, bw / max(bh, 1),
                    )
                    continue

            result.append(box)

        return result

    def _try_horizontal_split(
        self, page_image: Image.Image,
        x1: int, y1: int, x2: int, y2: int,
    ) -> Optional[List[Tuple[int, int, int, int]]]:
        """Try to split a tall box at its deepest horizontal whitespace gap.

        Returns two boxes if split succeeds, or None if it should be kept as-is.
        """
        bh = y2 - y1
        gray = np.array(page_image.crop((x1, y1, x2, y2)).convert("L"))
        dark = gray < self._INK_DARK_THRESHOLD
        row_density = np.mean(dark, axis=1).astype(np.float32)

        # Smooth
        kernel_size = 15
        kernel = np.ones(kernel_size) / kernel_size
        smoothed = np.convolve(row_density, kernel, mode="same")

        # Search in the middle 60% of the box
        margin = int(bh * 0.2)
        search_start = margin
        search_end = bh - margin
        if search_end <= search_start:
            return None

        region = smoothed[search_start:search_end]
        min_idx = int(np.argmin(region)) + search_start
        min_density = float(smoothed[min_idx])

        if min_density >= 0.02:
            return None

        split_y = y1 + min_idx
        box_a = (x1, y1, x2, split_y)
        box_b = (x1, split_y, x2, y2)

        # Both sub-boxes must have sufficient height
        if (split_y - y1 < self._MIN_STRUCTURE_ROW_HEIGHT
                or y2 - split_y < self._MIN_STRUCTURE_ROW_HEIGHT):
            return None

        # Both sub-boxes must pass structure ink validation
        if (self._has_structure_ink(page_image, *box_a)
                and self._has_structure_ink(page_image, *box_b)):
            return [box_a, box_b]

        return None

    def _try_vertical_split(
        self, page_image: Image.Image,
        x1: int, y1: int, x2: int, y2: int,
    ) -> Optional[List[Tuple[int, int, int, int]]]:
        """Try to split a wide box at its deepest vertical whitespace gap.

        Returns two boxes if split succeeds, or None if it should be kept as-is.
        """
        bw = x2 - x1
        bh = y2 - y1
        gray = np.array(page_image.crop((x1, y1, x2, y2)).convert("L"))
        dark = gray < self._INK_DARK_THRESHOLD
        col_density = np.mean(dark, axis=0).astype(np.float32)

        # Smooth
        kernel_size = 15
        kernel = np.ones(kernel_size) / kernel_size
        smoothed = np.convolve(col_density, kernel, mode="same")

        # Search in the middle 60%
        margin = int(bw * 0.2)
        search_start = margin
        search_end = bw - margin
        if search_end <= search_start:
            return None

        region = smoothed[search_start:search_end]
        min_idx = int(np.argmin(region)) + search_start
        min_density = float(smoothed[min_idx])

        if min_density >= 0.02:
            return None

        split_x = x1 + min_idx
        box_a = (x1, y1, split_x, y2)
        box_b = (split_x, y1, x2, y2)

        # Safety: both sub-boxes must have aspect ratio < _WIDE_SPLIT_MAX_ASPECT
        for bx1, by1, bx2, by2 in [box_a, box_b]:
            sub_w = bx2 - bx1
            sub_h = by2 - by1
            if sub_w < 10 or sub_h < 10:
                return None
            if sub_w / max(sub_h, 1) > self._WIDE_SPLIT_MAX_ASPECT:
                return None

        # Both sub-boxes must have structure ink
        if (self._has_structure_ink(page_image, *box_a)
                and self._has_structure_ink(page_image, *box_b)):
            return [box_a, box_b]

        return None

    @staticmethod
    def _box_inside_any_table(
        box: Tuple[int, int, int, int],
        table_boxes: List[Tuple[int, int, int, int]],
    ) -> bool:
        """Check if a box's center falls inside any table region."""
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        for tx1, ty1, tx2, ty2 in table_boxes:
            if tx1 <= cx <= tx2 and ty1 <= cy <= ty2:
                return True
        return False

    # ------------------------------------------------------------------
    # Ink and structure validation helpers
    # ------------------------------------------------------------------

    def _has_structure_ink(
        self,
        page_image: Image.Image,
        x1: int, y1: int, x2: int, y2: int,
    ) -> bool:
        """Check whether a region contains spatially-distributed ink (structure).

        Unlike simple ink detection, this divides the region into a 4x4 grid
        and counts cells with significant dark pixels. Structures spread ink
        across >=6 cells; text headers concentrate in 1-2 horizontal bands.
        """
        crop = page_image.crop((x1, y1, x2, y2)).convert("L")
        arr = np.array(crop)
        dark = arr < self._INK_DARK_THRESHOLD

        # First check overall ink threshold
        if dark.sum() / arr.size < self._INK_RATIO_THRESHOLD:
            return False

        h, w = arr.shape
        cell_h, cell_w = h // 4, w // 4
        if cell_h == 0 or cell_w == 0:
            return False

        occupied = sum(
            1 for r in range(4) for c in range(4)
            if dark[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w].sum()
               / (cell_h * cell_w) > self._STRUCTURE_INK_CELL_THRESHOLD
        )
        return occupied >= self._STRUCTURE_INK_MIN_CELLS

    def _template_similarity(
        self,
        page_image: Image.Image,
        candidate_box: Tuple[int, int, int, int],
    ) -> float:
        """Compute normalized cross-correlation between candidate and template."""
        candidate = np.array(
            page_image.crop(candidate_box).convert("L")
        )
        template = self._template_crop
        candidate_resized = cv2.resize(
            candidate, (template.shape[1], template.shape[0]),
        )
        result = cv2.matchTemplate(
            candidate_resized, template, cv2.TM_CCOEFF_NORMED,
        )
        return float(result[0][0])

    def _validate_candidate(
        self,
        page_image: Image.Image,
        x1: int, y1: int, x2: int, y2: int,
    ) -> bool:
        """Validate a synthetic candidate box using structure ink + template.

        Returns True if the region has spatially-distributed ink AND either
        no template is available or template similarity exceeds threshold.
        """
        if not self._has_structure_ink(page_image, x1, y1, x2, y2):
            return False

        if self._template_crop is not None:
            sim = self._template_similarity(page_image, (x1, y1, x2, y2))
            if sim < self._TEMPLATE_SIMILARITY_THRESHOLD:
                logger.debug(
                    "_validate_candidate: rejected (%d,%d,%d,%d) — similarity %.3f < %.3f",
                    x1, y1, x2, y2, sim, self._TEMPLATE_SIMILARITY_THRESHOLD,
                )
                return False

        return True

    def _has_ink(
        self,
        page_image: Image.Image,
        x1: int, y1: int, x2: int, y2: int,
    ) -> bool:
        """Check whether a region of the page image contains enough dark pixels."""
        crop = page_image.crop((x1, y1, x2, y2)).convert("L")
        arr = np.array(crop)
        dark_pixels = np.count_nonzero(arr < self._INK_DARK_THRESHOLD)
        total_pixels = arr.size
        if total_pixels == 0:
            return False
        return (dark_pixels / total_pixels) >= self._INK_RATIO_THRESHOLD
