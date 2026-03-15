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

    # Template matching — threshold is intentionally low because different
    # chemical structures have near-zero cross-correlation; we only want
    # to reject truly anti-correlated (inverted/blank) regions.
    _TEMPLATE_SIMILARITY_THRESHOLD = -0.3

    # Keywords for table caption scanning
    _TABLE_CAPTION_KEYWORDS = re.compile(
        r"(?:example|compound|structure|table\s*\d)", re.IGNORECASE,
    )

    def __init__(self, docling_classifier):
        """Initialize with a DoclingClassifier that has cached layout data.

        Args:
            docling_classifier: A DoclingClassifier instance that has already
                processed the PDF (layout clusters are cached).
        """
        self._classifier = docling_classifier
        self._table_row_cache = None  # (avg_h, x_left, x_right, pitch, header_offset)
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

    def _fill_table_gaps(
        self,
        boxes: List[Tuple[int, int, int, int]],
        page_image: Image.Image,
        page_num: int,
    ) -> List[Tuple[int, int, int, int]]:
        """Fill missed structure rows inside Docling-detected tables.

        When at least one structure box falls inside a table region, compute
        the expected row grid and synthesise candidate boxes for rows that
        have no detection.  The grid is anchored on detected structures and
        uses row pitch (distance between consecutive row starts) rather than
        structure height, since rows have inter-row gaps.

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

            # Compute average structure height and X-span
            heights = [boxes[i][3] - boxes[i][1] for i in inside_idxs]
            avg_h = sum(heights) / len(heights)
            if avg_h < 10:
                continue

            x_left = min(boxes[i][0] for i in inside_idxs)
            x_right = max(boxes[i][2] for i in inside_idxs)

            # Compute row pitch from consecutive detected y1 values
            sorted_y1s = sorted(boxes[i][1] for i in inside_idxs)

            if len(sorted_y1s) >= 2:
                intervals = [sorted_y1s[j + 1] - sorted_y1s[j]
                             for j in range(len(sorted_y1s) - 1)]
                pitch = sum(intervals) / len(intervals)
            elif self._table_row_cache is not None:
                pitch = self._table_row_cache[3]
            else:
                pitch = avg_h  # fallback when no cache and only 1 detection

            # Anchor on the first detected structure and walk up to find
            # the first data-row start within the table
            anchor_y = sorted_y1s[0]
            first_row_y = anchor_y
            while first_row_y - pitch >= ty1:
                first_row_y -= pitch

            header_offset = first_row_y - ty1

            # Cache for cross-page rescue
            self._table_row_cache = (avg_h, x_left, x_right, pitch, header_offset)
            self._template_crop = np.array(
                page_image.crop(boxes[inside_idxs[0]]).convert("L")
            )

            # Walk the grid from first_row_y by pitch
            y_cursor = first_row_y
            while y_cursor + avg_h <= ty2 + avg_h * 0.5:
                row_cy = y_cursor + avg_h / 2

                # Check if any existing box already covers this row center
                covered = False
                for bx1, by1, bx2, by2 in new_boxes:
                    if by1 <= row_cy <= by2 and bx1 <= (x_left + x_right) / 2 <= bx2:
                        covered = True
                        break

                if not covered:
                    cand_y1 = int(y_cursor)
                    cand_y2 = int(y_cursor + avg_h)

                    # Snap to the nearest detected box above: if a detected
                    # box ends just before this row, use its y2 as our y1
                    # so uneven pitch doesn't truncate the structure top.
                    for bx1, by1, bx2, by2 in new_boxes:
                        gap = cand_y1 - by2
                        if 0 < gap < pitch * 0.3:
                            cand_y1 = by2
                            break

                    # Extend to the next grid line or table bottom so we
                    # fill the full row rather than using rigid avg_h
                    next_grid = int(y_cursor + pitch)
                    cand_y2 = min(next_grid, ty2)

                    # Clamp to table and image boundaries
                    cand_y1 = max(0, max(cand_y1, ty1))
                    cand_y2 = min(page_image.size[1], cand_y2)
                    cand_x1 = max(0, max(int(x_left), tx1))
                    cand_x2 = min(page_image.size[0], min(int(x_right), tx2))

                    if cand_x2 - cand_x1 >= 10 and cand_y2 - cand_y1 >= 10:
                        if self._validate_candidate(
                            page_image, cand_x1, cand_y1, cand_x2, cand_y2,
                        ):
                            new_boxes.append((cand_x1, cand_y1, cand_x2, cand_y2))
                            logger.debug(
                                "_fill_table_gaps: added synthetic box (%d,%d,%d,%d) on page %d",
                                cand_x1, cand_y1, cand_x2, cand_y2, page_num,
                            )

                y_cursor += pitch

        if len(new_boxes) > len(boxes):
            logger.debug(
                "_fill_table_gaps: page %d — %d original + %d synthetic = %d total",
                page_num, len(boxes), len(new_boxes) - len(boxes), len(new_boxes),
            )

        return new_boxes

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
        2. Use cached row dimensions (or estimate from table size) to
           walk the table and synthesise candidate boxes
        3. Validate each candidate with _has_structure_ink + template matching

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

            # OCR a strip above the table to look for caption keywords
            caption_y1 = max(0, ty1 - 80)
            caption_y2 = ty1
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

            # Determine row dimensions
            table_h = ty2 - ty1
            if self._table_row_cache is not None:
                avg_h, x_left, x_right, cached_pitch, header_offset = self._table_row_cache
                # Adjust X-span to this table's bounds
                x_left = max(x_left, tx1)
                x_right = min(x_right, tx2)
                # Recalibrate pitch to this table's actual height to avoid
                # cumulative drift from cross-page dimension differences
                usable_h = table_h - header_offset
                est_rows = max(1, round(usable_h / cached_pitch))
                pitch = usable_h / est_rows
            else:
                # Standalone fallback: estimate from table dimensions
                header_offset = int(table_h * 0.07)
                usable_h = table_h - header_offset
                # Target ~400px per row at 200 DPI
                est_rows = max(2, min(6, round(usable_h / 400)))
                pitch = usable_h / est_rows
                avg_h = pitch * 0.9  # structures are ~90% of row pitch
                if avg_h < 10:
                    continue
                # Use full table width with small margins
                margin = int((tx2 - tx1) * 0.05)
                x_left = tx1 + margin
                x_right = tx2 - margin

            # Walk table Y-range starting after header
            y_cursor = ty1 + header_offset
            while y_cursor + avg_h <= ty2 + avg_h * 0.5:
                cand_y1 = int(y_cursor)
                next_grid = int(y_cursor + pitch)
                cand_y2 = min(next_grid, ty2)
                cand_x1 = max(0, int(x_left))
                cand_x2 = min(page_image.size[0], int(x_right))

                # Snap to nearest box above to avoid truncation
                for bx1, by1, bx2, by2 in new_boxes:
                    gap = cand_y1 - by2
                    if 0 < gap < pitch * 0.3:
                        cand_y1 = by2
                        break

                cand_y1 = max(0, cand_y1)
                cand_y2 = min(page_image.size[1], cand_y2)

                if cand_x2 - cand_x1 >= 10 and cand_y2 - cand_y1 >= 10:
                    # Check not already covered
                    row_cy = y_cursor + avg_h / 2
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

                y_cursor += pitch

        if len(new_boxes) > len(boxes):
            logger.debug(
                "_scan_example_tables: page %d — %d original + %d synthetic = %d total",
                page_num, len(boxes), len(new_boxes) - len(boxes), len(new_boxes),
            )

        return new_boxes

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
