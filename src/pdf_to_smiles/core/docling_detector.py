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
    PADDING_PX = 30

    # Inward margin from detected separator lines for candidate generation (Y-axis only).
    # Avoids including separator line pixels in validation region.
    _CELL_INWARD_MARGIN_Y = 8

    # Thresholds for _merge_split_boxes
    _MERGE_Y_OVERLAP_RATIO = 0.3
    _MERGE_X_GAP_PX = 50

    # Thresholds for ink detection
    _INK_DARK_THRESHOLD = 180  # grayscale value below which a pixel is "dark"
    _INK_RATIO_THRESHOLD = 0.01  # fraction of dark pixels to count as ink
    _STRUCTURE_INK_MIN_CELLS = 6  # min occupied cells in 4x4 grid
    _STRUCTURE_INK_CELL_THRESHOLD = 0.005  # min dark ratio per cell

    # Minimum row height for a candidate structure cell (px at 200 DPI).
    # Header rows are typically 40-110 px; real structure cells are 155+.
    # Set to 155 to catch compact table rows (e.g. GLP1 158px cells).
    _MIN_STRUCTURE_ROW_HEIGHT = 155

    # Minimum sub-box height after splitting a compound box. Slightly
    # higher than _MIN_STRUCTURE_ROW_HEIGHT to avoid over-splitting.
    _MIN_SPLIT_HEIGHT = 160

    # Template matching — threshold is intentionally low because different
    # chemical structures have near-zero cross-correlation; we only want
    # to reject truly anti-correlated (inverted/blank) regions.
    _TEMPLATE_SIMILARITY_THRESHOLD = -0.3

    # Adaptive row boundary anchoring thresholds
    _REFINE_SEARCH_ABOVE = 0          # px above boundary to search for example numbers
    _REFINE_SEARCH_BELOW = 50         # px below boundary to search (asymmetric: numbers are below gap center)
    _REFINE_MARGIN_ABOVE_INK = 5      # px margin above detected ink top for new boundary
    _REFINE_MIN_CONFIRMED = 2         # min boundaries with nearby ink to activate refinement
    _REFINE_MIN_CONFIRMED_RATIO = 0.45  # min fraction of internal boundaries confirmed
    _EXAMPLE_COL_WIDTH_RATIO = 0.15   # fraction of table width for example column heuristic
    _EXAMPLE_COL_MIN_PX = 60          # minimum example column width (px)
    _EXAMPLE_COL_MAX_PX = 150         # maximum example column width (px)

    # Tighten horizontal bounds via vertical ink density gap detection
    _TIGHTEN_GAP_DENSITY = 0.006       # max density to qualify as a vertical gap
    _TIGHTEN_GAP_MIN_WIDTH = 8         # min gap width in pixels
    _TIGHTEN_SEARCH_LEFT = 0.30        # right-gap search starts at 30% of table width
    _TIGHTEN_SEARCH_RIGHT = 0.80       # right-gap search ends at 80% of table width
    _TIGHTEN_MIN_RESULT_RATIO = 0.25   # tightened column must be >= 25% of table width
    _TIGHTEN_SMOOTH_KERNEL = 21        # smoothing kernel width for density profile

    # Ink extension — extend boxes vertically to capture truncated structure elements
    _INK_EXTEND_MAX_PX = 120           # max pixels to scan beyond box edge
    _INK_EXTEND_MIN_DENSITY = 0.008    # min ink density in a row to count as content
    _INK_EXTEND_GAP_ROWS = 8          # consecutive empty rows to stop extending

    # Connected component expansion — capture peripheral groups via ink connectivity
    _CC_EXPAND_SEARCH_PAD = 40         # px to pad search region beyond box
    _CC_EXPAND_CLOSE_KERNEL = 7        # morphological closing kernel (bridges ~5px gaps)
    _CC_EXPAND_MAX_PER_SIDE = 60       # max expansion per direction (px)
    _CC_EXPAND_MIN_COMPONENT = 50      # min component area (px) to consider

    # Grid columns narrower than this fraction of the full box width are
    # assumed to be label/number columns and skipped during decomposition.
    _GRID_MIN_COL_WIDTH_FRAC = 0.30

    # Keywords for table caption/header scanning.  "structure" is the primary
    # trigger; "compound" / "cpd" are secondary (guarded by validation).
    # Avoid "example" alone — it matches numeric data tables (TABLE 19 etc.).
    _TABLE_CAPTION_KEYWORDS = re.compile(
        r"structure|compound|cpd\.?\s*[#\d]",
        re.IGNORECASE,
    )

    def __init__(self, docling_classifier):
        """Initialize with a DoclingClassifier that has cached layout data.

        Args:
            docling_classifier: A DoclingClassifier instance that has already
                processed the PDF (layout clusters are cached).
        """
        self._classifier = docling_classifier
        self._table_row_cache = None  # List[(x_left, x_right)] — structure column X-ranges
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
        boxes = self._fill_column_gaps(boxes, page_image, page_num)
        boxes = self._split_compound_boxes(boxes, page_image, page_num)
        boxes = self._extend_boxes_to_ink(boxes, page_image)
        boxes = self._expand_boxes_by_connectivity(boxes, page_image)
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
    # Pre-processing: filter page-spanning layout artifacts
    # ------------------------------------------------------------------

    def _filter_page_spanning_boxes(
        self,
        boxes: List[Tuple[int, int, int, int]],
        page_num: int,
        page_image_size: Tuple[int, int],
    ) -> List[Tuple[int, int, int, int]]:
        """Remove raw boxes that span >70% of page height and aren't in tables.

        Docling sometimes labels full-page columns as "picture" — these are
        layout artifacts, not individual structures.  Boxes inside table
        regions are kept (they'll be decomposed into rows later).
        """
        if not boxes:
            return boxes

        img_w, img_h = page_image_size
        table_boxes = self._get_table_boxes(page_num, page_image_size)
        threshold_h = img_h * 0.7

        filtered = []
        for box in boxes:
            bh = box[3] - box[1]
            bw = box[2] - box[0]
            if bh > threshold_h and not self._box_inside_any_table(box, table_boxes):
                # Check if the box itself IS a table (same coords)
                is_table = any(
                    abs(box[0] - tb[0]) < 30 and abs(box[1] - tb[1]) < 30
                    and abs(box[2] - tb[2]) < 30 and abs(box[3] - tb[3]) < 30
                    for tb in table_boxes
                )
                # Keep wide boxes — they may be grids for decomposition
                is_wide = bw > 800
                if is_table or is_wide:
                    filtered.append(box)
                else:
                    logger.debug(
                        "_filter_page_spanning_boxes: removed (%d,%d,%d,%d) h=%d "
                        "(>70%% of page height %d)",
                        *box, bh, img_h,
                    )
            else:
                filtered.append(box)

        return filtered

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

    @staticmethod
    def _merge_short_row_segments(
        row_bounds: List[int], min_height: int,
    ) -> List[int]:
        """Merge row segments shorter than min_height with their successor.

        Tables with internal separator lines (e.g. between example number
        and structure drawing) create segments like [42px, 158px, 43px, 161px].
        This merges the short segments downward: remove the boundary between
        a short segment and the next one, combining them into one taller cell.
        The first and last boundaries are always preserved.
        """
        if len(row_bounds) < 3:
            return list(row_bounds)

        merged = [row_bounds[0]]
        i = 1
        while i < len(row_bounds):
            seg_h = row_bounds[i] - merged[-1]
            if seg_h < min_height and i < len(row_bounds) - 1:
                # Skip this boundary — merge this short segment with the next
                i += 1
                continue
            merged.append(row_bounds[i])
            i += 1
        return merged

    def _refine_row_boundaries(
        self,
        page_image: Image.Image,
        row_bounds: List[int],
        tx1: int, ty1: int, tx2: int, ty2: int,
        col_bounds: List[int],
    ) -> List[int]:
        """Shift internal row boundaries to just above example number ink.

        Patent tables have example numbers (e.g. "27", "64") in the leftmost
        column.  When row boundaries are placed at the center of whitespace
        gaps, structures whose content extends near the gap center get
        truncated.  This method detects the top edge of example number text
        near each boundary and shifts the boundary upward so the structure
        above gets maximum downward extent.

        Only activates when >=_REFINE_MIN_CONFIRMED boundaries have detectable
        ink in the example column strip (tables without example numbers are
        skipped).
        """
        if len(row_bounds) < 4:
            # Need at least 2 internal boundaries to validate
            return row_bounds

        # A. Determine example number column region
        table_w = tx2 - tx1
        if len(col_bounds) >= 3:
            ex_x1, ex_x2 = col_bounds[0], col_bounds[1]
        else:
            ex_w = min(self._EXAMPLE_COL_MAX_PX,
                       max(self._EXAMPLE_COL_MIN_PX,
                           int(table_w * self._EXAMPLE_COL_WIDTH_RATIO)))
            ex_x1, ex_x2 = tx1, tx1 + ex_w

        if ex_x2 - ex_x1 < 40:
            return row_bounds

        # B. Validate that example numbers exist — check ink near each
        #    internal boundary in the example column strip.
        #    Example numbers are compact text (~15-20px tall).  Reject if
        #    ink is spread across most of the check window (structure ink).
        confirmed = 0
        for i in range(1, len(row_bounds) - 1):
            b = row_bounds[i]
            check_y1 = max(ty1, b - 40)
            check_y2 = min(ty2, b + 40)
            if check_y2 - check_y1 < 10:
                continue
            crop = np.array(
                page_image.crop((ex_x1, check_y1, ex_x2, check_y2)).convert("L")
            )
            dark = crop < self._INK_DARK_THRESHOLD
            scanline_dens = np.mean(dark, axis=1)
            ink_lines = int(np.sum(scanline_dens > 0.005))
            window_h = check_y2 - check_y1
            # Must have some ink, but it should be compact (< 40% of
            # the window) — example numbers are ~15-20px, not 50+px.
            if 3 <= ink_lines <= int(window_h * 0.4):
                confirmed += 1

        n_internal = len(row_bounds) - 2
        ratio = confirmed / n_internal if n_internal > 0 else 0
        if (confirmed < self._REFINE_MIN_CONFIRMED
                or ratio < self._REFINE_MIN_CONFIRMED_RATIO):
            logger.debug(
                "_refine_row_boundaries: skipping — %d/%d (%.0f%%) boundaries "
                "have compact ink in example column",
                confirmed, n_internal, ratio * 100,
            )
            return row_bounds

        # C. For each internal boundary, find example number ink and shift
        refined = list(row_bounds)
        for i in range(1, len(refined) - 1):
            b = refined[i]
            search_y1 = max(ty1, b - self._REFINE_SEARCH_ABOVE)
            search_y2 = min(ty2, b + self._REFINE_SEARCH_BELOW)
            if search_y2 - search_y1 < 5:
                continue

            crop = np.array(
                page_image.crop((ex_x1, search_y1, ex_x2, search_y2)).convert("L")
            )
            dark = crop < self._INK_DARK_THRESHOLD
            scanline_density = np.mean(dark, axis=1)

            # Scan from top downward to find first scanline with ink
            ink_top_rel = None
            for j, d in enumerate(scanline_density):
                if d > 0.005:
                    ink_top_rel = j
                    break

            if ink_top_rel is None:
                continue

            ink_top_abs = search_y1 + ink_top_rel
            new_boundary = ink_top_abs - self._REFINE_MARGIN_ABOVE_INK

            # Clamp to maintain minimum 20px between boundaries
            lower_clamp = refined[i - 1] + 20
            upper_clamp = refined[i + 1] - 20
            new_boundary = max(lower_clamp, min(upper_clamp, new_boundary))

            if new_boundary != refined[i]:
                logger.debug(
                    "_refine_row_boundaries: boundary %d shifted %d → %d (Δ=%d)",
                    i, refined[i], new_boundary, new_boundary - refined[i],
                )
                refined[i] = new_boundary

        return refined

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

    def _identify_structure_columns(
        self, col_boundaries: List[int],
        boxes: List[Tuple], inside_idxs: List[int],
    ) -> List[Tuple[int, int]]:
        """Return list of (x_left, x_right) for all structure columns.

        Identifies every column that contains the center of at least one
        detected structure box.  Falls back to cached columns, and finally
        to a heuristic (all columns except the narrowest / first).
        """
        if len(col_boundaries) < 2:
            return [(col_boundaries[0], col_boundaries[-1])] if col_boundaries else []

        # Build list of column ranges
        col_ranges = [
            (col_boundaries[k], col_boundaries[k + 1])
            for k in range(len(col_boundaries) - 1)
        ]

        if inside_idxs:
            mid_xs = [(boxes[i][0] + boxes[i][2]) / 2 for i in inside_idxs]
            hit_cols = set()
            for mx in mid_xs:
                for k, (cl, cr) in enumerate(col_ranges):
                    if cl <= mx <= cr:
                        hit_cols.add(k)
                        break
            if hit_cols:
                return [col_ranges[k] for k in sorted(hit_cols)]

        # Fallback: use cached column ranges
        if self._table_row_cache is not None:
            matched = []
            for cached_xl, cached_xr in self._table_row_cache:
                cached_mid = (cached_xl + cached_xr) / 2
                for cl, cr in col_ranges:
                    if cl <= cached_mid <= cr:
                        if (cl, cr) not in matched:
                            matched.append((cl, cr))
                        break
            if matched:
                return matched

        # Conservative fallback: second column only (patent tables are
        # typically Example | Structure | Name | ...).
        if len(col_ranges) >= 2:
            return [col_ranges[1]]

        # Last resort: full table width
        return [(col_boundaries[0], col_boundaries[-1])]

    def _filter_narrow_columns(
        self,
        struct_cols: List[Tuple[int, int]],
        table_width: int,
    ) -> List[Tuple[int, int]]:
        """Remove columns narrower than 30% of total width (label/number columns)."""
        if len(struct_cols) <= 1:
            return struct_cols
        filtered = [
            (xl, xr) for xl, xr in struct_cols
            if (xr - xl) / max(table_width, 1) >= self._GRID_MIN_COL_WIDTH_FRAC
        ]
        return filtered if filtered else struct_cols  # keep at least one

    def _tighten_struct_column_bounds(
        self,
        page_image: Image.Image,
        struct_cols: List[Tuple[int, int]],
        row_bounds: List[int],
        tx1: int, ty1: int, tx2: int, ty2: int,
        col_bounds: List[int],
        existing_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> List[Tuple[int, int]]:
        """Tighten horizontal bounds by finding vertical ink density gaps.

        When a table has no internal vertical separator lines (col_bounds < 3),
        struct_cols spans the full table width. This method looks for a clear
        vertical gap in ink density (e.g. between structure drawings and NMR
        text or IUPAC names) and narrows the column accordingly.
        """
        table_w = tx2 - tx1

        # Guard: columns already detected by line detection
        if len(col_bounds) >= 3:
            return struct_cols

        # Guard: multi-column or empty
        if len(struct_cols) != 1:
            return struct_cols

        # Guard: already narrow enough (< 60% of table width)
        sc_left, sc_right = struct_cols[0]
        if (sc_right - sc_left) < table_w * 0.60:
            return struct_cols

        # Guard: not enough data rows to sample
        if len(row_bounds) < 3:
            return struct_cols

        # Build composite vertical ink density profile across data rows
        # (skip header row at index 0)
        profiles = []
        for r in range(1, len(row_bounds) - 1):
            ry1 = row_bounds[r]
            ry2 = row_bounds[r + 1]
            if ry2 - ry1 < 10:
                continue
            row_crop = page_image.crop((tx1, ry1, tx2, ry2)).convert("L")
            row_arr = np.array(row_crop, dtype=np.float32)
            # Dark pixel density per column
            dark = (row_arr < self._INK_DARK_THRESHOLD).astype(np.float32)
            col_density = dark.mean(axis=0)  # shape: (table_w,)
            profiles.append(col_density)

        if not profiles:
            return struct_cols

        # Average across all sampled rows
        avg_profile = np.mean(profiles, axis=0)

        # Smooth with uniform kernel
        k = self._TIGHTEN_SMOOTH_KERNEL
        if len(avg_profile) > k:
            kernel = np.ones(k) / k
            avg_profile = np.convolve(avg_profile, kernel, mode="same")

        # --- Right boundary: find leftmost qualifying gap in [30%, 80%] ---
        search_l = int(table_w * self._TIGHTEN_SEARCH_LEFT)
        search_r = int(table_w * self._TIGHTEN_SEARCH_RIGHT)
        new_x2 = sc_right  # default: unchanged

        gap_start = None
        for i in range(search_l, min(search_r, len(avg_profile))):
            if avg_profile[i] < self._TIGHTEN_GAP_DENSITY:
                if gap_start is None:
                    gap_start = i
                if i - gap_start + 1 >= self._TIGHTEN_GAP_MIN_WIDTH:
                    new_x2 = tx1 + gap_start
                    break
            else:
                gap_start = None

        # --- Left boundary: find rightmost qualifying gap in [5%, 25%] ---
        left_search_l = int(table_w * 0.05)
        left_search_r = int(table_w * 0.25)
        new_x1 = sc_left  # default: unchanged

        gap_start = None
        for i in range(left_search_l, min(left_search_r, len(avg_profile))):
            if avg_profile[i] < self._TIGHTEN_GAP_DENSITY:
                if gap_start is None:
                    gap_start = i
                gap_end_candidate = i
            else:
                if gap_start is not None:
                    if gap_end_candidate - gap_start + 1 >= self._TIGHTEN_GAP_MIN_WIDTH:
                        new_x1 = tx1 + gap_end_candidate + 1
                gap_start = None
        # Check last run
        if gap_start is not None:
            if gap_end_candidate - gap_start + 1 >= self._TIGHTEN_GAP_MIN_WIDTH:
                new_x1 = tx1 + gap_end_candidate + 1

        # Validate: tightened width must be >= 25% of table width
        if (new_x2 - new_x1) < table_w * self._TIGHTEN_MIN_RESULT_RATIO:
            return struct_cols

        # Only return if we actually tightened something
        if new_x1 == sc_left and new_x2 == sc_right:
            return struct_cols

        # Validate against existing Docling-detected boxes: if any existing
        # box extends significantly beyond a proposed tightened bound,
        # revert that individual bound (the structures occupy that area).
        if existing_boxes:
            for bx1, by1, bx2, by2 in existing_boxes:
                bcx = (bx1 + bx2) / 2
                bcy = (by1 + by2) / 2
                if not (tx1 <= bcx <= tx2 and ty1 <= bcy <= ty2):
                    continue
                if bx2 > new_x2 + 20:
                    logger.debug(
                        "_tighten_struct_column_bounds: reverting right — box x2=%d > proposed %d",
                        bx2, new_x2,
                    )
                    new_x2 = sc_right
                if bx1 < new_x1 - 20:
                    logger.debug(
                        "_tighten_struct_column_bounds: reverting left — box x1=%d < proposed %d",
                        bx1, new_x1,
                    )
                    new_x1 = sc_left

        logger.debug(
            "_tighten_struct_column_bounds: [%d, %d] -> [%d, %d] (table width %d)",
            sc_left, sc_right, new_x1, new_x2, table_w,
        )
        return [(new_x1, new_x2)]

    def _extend_boxes_to_ink(
        self,
        boxes: List[Tuple[int, int, int, int]],
        page_image: Image.Image,
    ) -> List[Tuple[int, int, int, int]]:
        """Extend boxes vertically to capture structure ink beyond boundaries.

        Scans narrow bands above and below each box for dark pixels that
        continue from the structure content.  Only extends when the band
        is not already covered by another box.  Limits extension to
        ``_INK_EXTEND_MAX_PX`` and stops at the first gap of
        ``_INK_EXTEND_GAP_ROWS`` consecutive empty rows.
        """
        if not boxes:
            return boxes

        gray = np.array(page_image.convert("L"))
        img_h, img_w = gray.shape
        max_ext = self._INK_EXTEND_MAX_PX
        min_den = self._INK_EXTEND_MIN_DENSITY
        gap_tol = self._INK_EXTEND_GAP_ROWS

        result = list(boxes)

        for i in range(len(result)):
            x1, y1, x2, y2 = result[i]

            # Narrow the X scan range to where ink actually exists inside
            # the box.  Wide table-fill boxes dilute the density when the
            # structure only occupies part of the width.
            box_region = gray[y1:y2, x1:x2]
            col_dark = (box_region < self._INK_DARK_THRESHOLD).mean(axis=0)
            ink_cols = np.where(col_dark > 0.005)[0]
            if len(ink_cols) > 2:
                sx1 = x1 + int(ink_cols[0])
                sx2 = x1 + int(ink_cols[-1]) + 1
            else:
                sx1, sx2 = x1, x2

            # --- Extend ABOVE ---
            scan_top = max(0, y1 - max_ext)
            if scan_top < y1:
                band = gray[scan_top:y1, sx1:sx2]
                dark = (band < self._INK_DARK_THRESHOLD).astype(np.float32)
                new_y1 = y1
                consecutive_empty = 0
                # Scan bottom-to-top (closest to box first)
                for row in range(dark.shape[0] - 1, -1, -1):
                    if dark[row].mean() >= min_den:
                        new_y1 = scan_top + row
                        consecutive_empty = 0
                    else:
                        consecutive_empty += 1
                        if consecutive_empty >= gap_tol and new_y1 < y1:
                            break
                if new_y1 < y1:
                    result[i] = (x1, new_y1, x2, y2)
                    logger.debug(
                        "_extend_boxes_to_ink: box %d extended UP by %d px",
                        i, y1 - new_y1,
                    )

            # Refresh after possible above-extension
            x1, y1, x2, y2 = result[i]

            # --- Extend BELOW ---
            scan_bot = min(img_h, y2 + max_ext)
            if scan_bot > y2:
                band = gray[y2:scan_bot, sx1:sx2]
                dark = (band < self._INK_DARK_THRESHOLD).astype(np.float32)
                new_y2 = y2
                consecutive_empty = 0
                # Scan top-to-bottom (closest to box first)
                for row in range(dark.shape[0]):
                    if dark[row].mean() >= min_den:
                        new_y2 = y2 + row + 1
                        consecutive_empty = 0
                    else:
                        consecutive_empty += 1
                        if consecutive_empty >= gap_tol and new_y2 > y2:
                            break
                if new_y2 > y2:
                    result[i] = (x1, y1, x2, new_y2)
                    logger.debug(
                        "_extend_boxes_to_ink: box %d extended DOWN by %d px",
                        i, new_y2 - y2,
                    )

        return result

    def _expand_boxes_by_connectivity(
        self,
        boxes: List[Tuple[int, int, int, int]],
        page_image: Image.Image,
    ) -> List[Tuple[int, int, int, int]]:
        """Expand boxes to include ink blobs connected to box content.

        After ink-density extension, peripheral functional groups (e.g.
        isopropyl H3C/CH3, methoxyethyl chains) may still be clipped
        because their ink is sparse relative to box width.  This method
        uses morphological closing + connected component analysis to find
        ink regions physically connected to each box's content and expands
        the box to encompass them.
        """
        if not boxes:
            return boxes

        gray = np.array(page_image.convert("L"))
        img_h, img_w = gray.shape
        pad = self._CC_EXPAND_SEARCH_PAD
        max_exp = self._CC_EXPAND_MAX_PER_SIDE
        min_area = self._CC_EXPAND_MIN_COMPONENT
        k_size = self._CC_EXPAND_CLOSE_KERNEL

        # Precompute center points of all boxes for collision guard
        centers = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]

        result = list(boxes)

        for i in range(len(result)):
            x1, y1, x2, y2 = result[i]

            # 1. Search region: pad box, clamp to image bounds
            sx1 = max(0, x1 - pad)
            sy1 = max(0, y1 - pad)
            sx2 = min(img_w, x2 + pad)
            sy2 = min(img_h, y2 + pad)

            # 2. Binarize search region
            region = gray[sy1:sy2, sx1:sx2]
            binary = (region < self._INK_DARK_THRESHOLD).astype(np.uint8) * 255

            # 3. Morphological closing to bridge bond-to-label gaps
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (k_size, k_size),
            )
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            # 4. Connected components
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                closed, connectivity=8,
            )

            # 5. Find components overlapping the original box region
            # Box coordinates in search-region space
            bx1_local = x1 - sx1
            by1_local = y1 - sy1
            bx2_local = x2 - sx1
            by2_local = y2 - sy1

            # Create mask of the box region in search-region coords
            box_mask = np.zeros(labels.shape, dtype=bool)
            box_mask[by1_local:by2_local, bx1_local:bx2_local] = True

            # Find labels that overlap the box
            overlapping_labels = set(np.unique(labels[box_mask])) - {0}

            # 6. Build expanded bounds from qualifying components
            exp_x1, exp_y1, exp_x2, exp_y2 = x1, y1, x2, y2
            found_expansion = False

            for lbl in range(1, n_labels):
                if lbl not in overlapping_labels:
                    continue
                area = int(stats[lbl, cv2.CC_STAT_AREA])
                if area < min_area:
                    continue

                # Component bounding rect in search-region coords
                cx = int(stats[lbl, cv2.CC_STAT_LEFT])
                cy = int(stats[lbl, cv2.CC_STAT_TOP])
                cw = int(stats[lbl, cv2.CC_STAT_WIDTH])
                ch = int(stats[lbl, cv2.CC_STAT_HEIGHT])

                # Convert to image coords
                comp_x1 = sx1 + cx
                comp_y1 = sy1 + cy
                comp_x2 = sx1 + cx + cw
                comp_y2 = sy1 + cy + ch

                exp_x1 = min(exp_x1, comp_x1)
                exp_y1 = min(exp_y1, comp_y1)
                exp_x2 = max(exp_x2, comp_x2)
                exp_y2 = max(exp_y2, comp_y2)
                found_expansion = True

            if not found_expansion:
                continue

            # 7. Cap expansion per direction
            exp_x1 = max(exp_x1, x1 - max_exp)
            exp_y1 = max(exp_y1, y1 - max_exp)
            exp_x2 = min(exp_x2, x2 + max_exp)
            exp_y2 = min(exp_y2, y2 + max_exp)

            # Clamp to image bounds
            exp_x1 = max(0, exp_x1)
            exp_y1 = max(0, exp_y1)
            exp_x2 = min(img_w, exp_x2)
            exp_y2 = min(img_h, exp_y2)

            # 8. Collision guard: revert expansion in any direction if it
            # would encompass another box's center
            for j, (cx, cy) in enumerate(centers):
                if j == i:
                    continue
                # Check if the other box's center falls inside the expanded box
                if exp_x1 <= cx <= exp_x2 and exp_y1 <= cy <= exp_y2:
                    # Revert the direction(s) that caused the collision
                    ox1, oy1, ox2, oy2 = result[i]
                    if cx < ox1:  # other center is to the left
                        exp_x1 = ox1
                    if cx > ox2:  # other center is to the right
                        exp_x2 = ox2
                    if cy < oy1:  # other center is above
                        exp_y1 = oy1
                    if cy > oy2:  # other center is below
                        exp_y2 = oy2

            if (exp_x1, exp_y1, exp_x2, exp_y2) != (x1, y1, x2, y2):
                logger.debug(
                    "_expand_boxes_by_connectivity: box %d expanded "
                    "(%d,%d,%d,%d) → (%d,%d,%d,%d)",
                    i, x1, y1, x2, y2, exp_x1, exp_y1, exp_x2, exp_y2,
                )
                result[i] = (exp_x1, exp_y1, exp_x2, exp_y2)

        return result

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

            # Extend table bottom if last boundary is suspiciously close to
            # the Docling table edge — gives the last row room to breathe.
            img_h = page_image.size[1]
            if len(row_bounds) >= 2:
                last_b = row_bounds[-1]
                if last_b - row_bounds[-2] < self._MIN_STRUCTURE_ROW_HEIGHT:
                    extend = min(50, img_h - last_b)
                    if extend > 0:
                        row_bounds[-1] = last_b + extend

            # Identify structure columns (may be >1 for multi-column tables)
            struct_cols = self._identify_structure_columns(
                col_bounds, boxes, inside_idxs,
            )
            struct_cols = self._filter_narrow_columns(struct_cols, tx2 - tx1)

            # Tighten horizontal bounds via vertical ink density gap detection
            inside_boxes = [boxes[i] for i in inside_idxs]
            struct_cols = self._tighten_struct_column_bounds(
                page_image, struct_cols, row_bounds,
                tx1, ty1, tx2, ty2, col_bounds,
                existing_boxes=inside_boxes,
            )

            # Cache structure column X-ranges for cross-page rescue
            self._table_row_cache = struct_cols
            self._template_crop = np.array(
                page_image.crop(boxes[inside_idxs[0]]).convert("L")
            )

            # Refine row boundaries by anchoring just above example numbers
            row_bounds = self._refine_row_boundaries(
                page_image, row_bounds, tx1, ty1, tx2, ty2, col_bounds,
            )

            # Detect oversized boxes: if any inside box covers >80% of the
            # table height and we have ≥3 row boundaries, remove it so it
            # gets decomposed into per-row candidates below.
            table_h = ty2 - ty1
            if len(row_bounds) >= 3:
                oversized = set()
                for idx in inside_idxs:
                    bh = boxes[idx][3] - boxes[idx][1]
                    if bh > table_h * 0.8:
                        oversized.add(boxes[idx])
                if oversized:
                    new_boxes = [b for b in new_boxes if b not in oversized]
                    logger.debug(
                        "_fill_table_gaps: page %d decomposing %d oversized boxes into rows",
                        page_num, len(oversized),
                    )

            # Skip the header row (first segment) — data rows start from
            # the second row segment onward
            data_start = 1 if len(row_bounds) > 2 else 0

            for r in range(data_start, len(row_bounds) - 1):
                raw_row_h = row_bounds[r + 1] - row_bounds[r]
                cand_y1 = row_bounds[r] + self._CELL_INWARD_MARGIN_Y
                cand_y2 = row_bounds[r + 1] - self._CELL_INWARD_MARGIN_Y
                is_last_row = (r == len(row_bounds) - 2)

                for x_left, x_right in struct_cols:
                    cand_x1 = max(0, x_left)
                    cand_x2 = min(page_image.size[0], x_right)

                    if cand_x2 - cand_x1 < 10:
                        continue

                    # Apply relaxed height threshold for last row (Fix 3)
                    # Use raw row height (before inward margin) for threshold check
                    min_h = self._MIN_STRUCTURE_ROW_HEIGHT
                    if is_last_row and raw_row_h >= 100:
                        min_h = 100
                    if raw_row_h < min_h:
                        continue

                    # Check if any existing box already covers this cell
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

            keyword_matched = bool(self._TABLE_CAPTION_KEYWORDS.search(caption_text))
            # Allow cache-based rescue for continuation pages: if the cache
            # exists and its column range overlaps the table, proceed even
            # without keyword match.
            cache_compatible = False
            if not keyword_matched and self._table_row_cache is not None:
                for cached_xl, cached_xr in self._table_row_cache:
                    if cached_xl >= tx1 and cached_xr <= tx2:
                        cache_compatible = True
                        break
            if not keyword_matched and not cache_compatible:
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

            # Identify structure columns
            struct_cols = self._identify_structure_columns(
                col_bounds, boxes, [],
            )
            struct_cols = self._filter_narrow_columns(struct_cols, tx2 - tx1)

            # When few vertical separator lines exist (col_bounds < 3),
            # struct_cols is the full table width. Use cached tightened
            # bounds if genuinely narrower; apply margin fallback only
            # when no cache exists (preserving baseline behavior).
            if len(col_bounds) < 3:
                table_w = tx2 - tx1
                if self._table_row_cache is not None and len(self._table_row_cache) == 1:
                    c_left, c_right = self._table_row_cache[0]
                    cache_w = c_right - c_left
                    # Use cache if genuinely narrower (< 80% of table) and fits
                    if cache_w < table_w * 0.80 and c_left >= tx1 and c_right <= tx2:
                        struct_cols = list(self._table_row_cache)
                elif self._table_row_cache is None:
                    margin = int(table_w * 0.05)
                    struct_cols = [(tx1 + margin, tx2 - margin)]

            # Cache structure column X-ranges for subsequent pages
            self._table_row_cache = struct_cols

            # Refine row boundaries by anchoring just above example numbers
            row_bounds = self._refine_row_boundaries(
                page_image, row_bounds, tx1, ty1, tx2, ty2, col_bounds,
            )

            # Skip the header row
            data_start = 1 if len(row_bounds) > 2 else 0

            # Collect candidates from all structure columns
            table_candidates = []
            for r in range(data_start, len(row_bounds) - 1):
                raw_row_h = row_bounds[r + 1] - row_bounds[r]
                cand_y1 = row_bounds[r] + self._CELL_INWARD_MARGIN_Y
                cand_y2 = row_bounds[r + 1] - self._CELL_INWARD_MARGIN_Y
                is_last_row = (r == len(row_bounds) - 2)

                for x_left, x_right in struct_cols:
                    cand_x1 = max(0, x_left)
                    cand_x2 = min(page_image.size[0], x_right)

                    if cand_x2 - cand_x1 < 10:
                        continue

                    # Apply relaxed height threshold for last row (Fix 3)
                    # Use raw row height (before inward margin) for threshold check
                    min_h = self._MIN_STRUCTURE_ROW_HEIGHT
                    if is_last_row and raw_row_h >= 100:
                        min_h = 100
                    if raw_row_h < min_h:
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
                        table_candidates.append((cand_x1, cand_y1, cand_x2, cand_y2))

            # Guard: cache-based rescue (no keyword match) requires ≥2
            # validated candidates to avoid false positives from text tables.
            min_candidates = 2 if (not keyword_matched and cache_compatible) else 1
            if len(table_candidates) >= min_candidates:
                new_boxes.extend(table_candidates)
                for cand in table_candidates:
                    logger.debug(
                        "_scan_example_tables: added synthetic box (%d,%d,%d,%d) on page %d",
                        *cand, page_num,
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

            # For each consecutive pair, close the gap (upward)
            for k in range(1, len(inside)):
                prev_idx = inside[k - 1]
                curr_idx = inside[k]
                prev_y2 = result[prev_idx][3]
                curr_x1, curr_y1, curr_x2, curr_y2 = result[curr_idx]
                gap = curr_y1 - prev_y2

                if 0 < gap < 80:  # small gap likely containing example number
                    result[curr_idx] = (curr_x1, prev_y2, curr_x2, curr_y2)

            # Also extend boxes downward to close gaps below
            for k in range(len(inside) - 1):
                curr_idx = inside[k]
                next_idx = inside[k + 1]
                curr_x1, curr_y1, curr_x2, curr_y2 = result[curr_idx]
                next_y1 = result[next_idx][1]
                gap = next_y1 - curr_y2

                if 0 < gap < 80:
                    result[curr_idx] = (curr_x1, curr_y1, curr_x2, next_y1)

        return result

    # ------------------------------------------------------------------
    # Post-processing: fill gaps in multi-column layouts
    # ------------------------------------------------------------------

    def _fill_column_gaps(
        self,
        boxes: List[Tuple[int, int, int, int]],
        page_image: Image.Image,
        page_num: int,
    ) -> List[Tuple[int, int, int, int]]:
        """Fill missing positions in a regular multi-column grid layout.

        When ≥6 boxes form a 2-column grid with regular vertical pitch,
        check for missing grid positions and add validated candidates.
        Only runs on pages without table regions (i.e., Docling detected
        the structures as individual boxes, not as a table).
        """
        table_boxes = self._get_table_boxes(page_num, page_image.size)
        if table_boxes or len(boxes) < 6:
            return boxes

        img_w = page_image.size[0]
        mid_x = img_w / 2

        # Split into left/right columns
        left = sorted([b for b in boxes if (b[0] + b[2]) / 2 < mid_x],
                       key=lambda b: b[1])
        right = sorted([b for b in boxes if (b[0] + b[2]) / 2 >= mid_x],
                        key=lambda b: b[1])

        if len(left) < 2 or len(right) < 2:
            return boxes

        # Compute average box dimensions and pitch for each column
        new_boxes = list(boxes)

        for col_boxes in [left, right]:
            avg_w = int(sum(b[2] - b[0] for b in col_boxes) / len(col_boxes))
            avg_h = int(sum(b[3] - b[1] for b in col_boxes) / len(col_boxes))
            avg_x1 = int(sum(b[0] for b in col_boxes) / len(col_boxes))
            avg_x2 = avg_x1 + avg_w

            if len(col_boxes) < 2:
                continue

            # Compute pitch from consecutive boxes
            pitches = [col_boxes[i + 1][1] - col_boxes[i][1]
                       for i in range(len(col_boxes) - 1)]
            avg_pitch = sum(pitches) / len(pitches)
            if avg_pitch < 100:
                continue

            # Check for gaps: walk from top of other column to bottom
            # and look for missing positions
            other = right if col_boxes is left else left
            if not other:
                continue

            # Try to find a missing position above the first box
            first_y = col_boxes[0][1]
            if other and other[0][1] < first_y - avg_pitch * 0.5:
                # There might be a row above the first detection
                cand_y1 = first_y - int(avg_pitch)
                cand_y2 = cand_y1 + avg_h
                if cand_y1 >= 0 and cand_y2 > cand_y1:
                    if self._validate_candidate(
                        page_image, avg_x1, cand_y1, avg_x2, cand_y2,
                    ):
                        new_boxes.append((avg_x1, cand_y1, avg_x2, cand_y2))
                        logger.debug(
                            "_fill_column_gaps: added (%d,%d,%d,%d) on page %d",
                            avg_x1, cand_y1, avg_x2, cand_y2, page_num,
                        )

            # Check for gaps between consecutive boxes
            for i in range(len(col_boxes) - 1):
                gap = col_boxes[i + 1][1] - col_boxes[i][3]
                if gap > avg_pitch * 0.5:
                    cand_y1 = col_boxes[i][3]
                    cand_y2 = cand_y1 + avg_h
                    if self._validate_candidate(
                        page_image, avg_x1, cand_y1, avg_x2, cand_y2,
                    ):
                        new_boxes.append((avg_x1, cand_y1, avg_x2, cand_y2))
                        logger.debug(
                            "_fill_column_gaps: added (%d,%d,%d,%d) on page %d",
                            avg_x1, cand_y1, avg_x2, cand_y2, page_num,
                        )

        if len(new_boxes) > len(boxes):
            logger.debug(
                "_fill_column_gaps: page %d — added %d boxes",
                page_num, len(new_boxes) - len(boxes),
            )

        return new_boxes

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

            # --- Case 0: Large grid box → decompose into grid cells ---
            # A box that's both wide (>800px) and tall (>500px) and not
            # inside a table may be a grid of small structures.
            if (bw > 800 and bh > 500
                    and not self._box_inside_any_table(box, table_boxes)):
                grid_cells = self._try_grid_decompose(
                    page_image, x1, y1, x2, y2,
                )
                if grid_cells is not None:
                    result.extend(grid_cells)
                    logger.debug(
                        "_split_compound_boxes: page %d grid box (%d,%d,%d,%d) "
                        "→ decomposed into %d cells",
                        page_num, x1, y1, x2, y2, len(grid_cells),
                    )
                    continue

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
        if (split_y - y1 < self._MIN_SPLIT_HEIGHT
                or y2 - split_y < self._MIN_SPLIT_HEIGHT):
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

    def _try_grid_decompose(
        self, page_image: Image.Image,
        x1: int, y1: int, x2: int, y2: int,
    ) -> Optional[List[Tuple[int, int, int, int]]]:
        """Decompose a large box into grid cells using line detection.

        For boxes that contain a grid of small structures (e.g. intermediate
        tables in patent documents), detect internal row/column separators
        and generate individual cell candidates.

        Returns a list of validated cell boxes, or None if the box doesn't
        appear to be a grid (fewer than 2 columns or 2 rows detected).
        """
        row_bounds = self._detect_table_row_boundaries(
            page_image, x1, y1, x2, y2,
        )
        col_bounds = self._detect_table_column_boundaries(
            page_image, x1, y1, x2, y2,
        )

        # Need at least a 2x2 grid to consider this a grid box
        if len(row_bounds) < 3 or len(col_bounds) < 3:
            return None

        cells = []
        # Don't skip first row — the grid box typically starts below headers
        box_w = x2 - x1

        # Pre-compute which columns to keep.  Narrow label/number columns
        # (< 30% of box width) are dropped only when wider columns exist;
        # if ALL columns are narrow (e.g. a 4-column equal grid) keep them.
        col_ranges = [
            (col_bounds[c], col_bounds[c + 1])
            for c in range(len(col_bounds) - 1)
            if col_bounds[c + 1] - col_bounds[c] >= 80
        ]
        wide_cols = [
            (cx1, cx2) for cx1, cx2 in col_ranges
            if (cx2 - cx1) / max(box_w, 1) >= self._GRID_MIN_COL_WIDTH_FRAC
        ]
        keep_cols = wide_cols if wide_cols else col_ranges

        for r in range(len(row_bounds) - 1):
            ry1 = row_bounds[r] + self._CELL_INWARD_MARGIN_Y
            ry2 = row_bounds[r + 1] - self._CELL_INWARD_MARGIN_Y
            if ry2 - ry1 < 100:
                continue

            for cx1, cx2 in keep_cols:

                if self._validate_candidate(page_image, cx1, ry1, cx2, ry2):
                    cells.append((cx1, ry1, cx2, ry2))

        # Only return grid cells if we found a reasonable number
        if len(cells) >= 4:
            return cells
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
