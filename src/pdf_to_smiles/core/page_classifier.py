"""Fast visual page classifier for detecting pages with chemical structures or bio data.

Uses pixel-level analysis at 72 DPI — no OCR, no ML. Designed to quickly scan
100+ page patent PDFs and identify the ~5-15 pages containing structures/tables.
"""

from typing import List

import pypdfium2 as pdfium
from PIL import Image


def get_classifier():
    """Return the best available page classifier.

    Priority chain:
    1. Hybrid (Docling + Claude Vision verification) -- best accuracy
    2. Docling standalone -- good accuracy, free, Apache 2.0
    3. Heuristic PageClassifier -- basic fallback
    """
    try:
        from .docling_classifier import DoclingClassifier
        docling_available = True
    except ImportError:
        docling_available = False

    try:
        from .llm_layout_analyzer import is_available as llm_available
        llm_ok = llm_available()
    except ImportError:
        llm_ok = False

    if docling_available and llm_ok:
        from .hybrid_classifier import HybridClassifier
        return HybridClassifier()
    elif docling_available:
        from .docling_classifier import DoclingClassifier
        return DoclingClassifier()
    else:
        return PageClassifier()


class PageClassifier:
    """Classify PDF pages by visual content using low-resolution pixel analysis."""

    # Rendering
    DPI = 72  # Low res for speed (~600x800 px per page)

    # Thresholds for structure detection
    DARK_THRESHOLD = 200        # Grayscale value below which pixels are "dark"
    STRIP_HEIGHT = 40           # Horizontal strip height in pixels
    MIN_DARK_RATIO = 0.005      # Minimum dark pixel ratio in a strip to consider
    MAX_SPREAD_RATIO = 0.80     # Max spread ratio for "localized" (not full-width text)
    MIN_STRUCTURE_STRIPS = 2    # Minimum strips with localized dark clusters
    MIN_CLUSTER_SIZE = 80       # Minimum contiguous dark region size (px) at 72 DPI

    # Thresholds for bio table detection
    TABLE_GRID_MIN_COLS = 3     # Minimum column-like vertical bands
    TABLE_GRID_MIN_ROWS = 4    # Minimum row-like horizontal bands

    def detect_structure_pages(
        self,
        pdf_path: str,
        progress_callback=None,
    ) -> List[int]:
        """Scan PDF and return 1-indexed page numbers likely containing structures or bio data.

        Args:
            pdf_path: Path to the PDF file.
            progress_callback: Optional callable(current_page, total_pages) for progress.

        Returns:
            Sorted list of 1-indexed page numbers.
        """
        detected = []

        doc = pdfium.PdfDocument(pdf_path)
        total_pages = len(doc)

        for page_idx in range(total_pages):
            if progress_callback:
                progress_callback(page_idx + 1, total_pages)

            page = doc[page_idx]
            bitmap = page.render(scale=self.DPI / 72)
            pil_image = bitmap.to_pil()

            if self._has_structure_graphics(pil_image) or self._has_bio_table_indicators(pil_image):
                detected.append(page_idx + 1)  # 1-indexed

        doc.close()
        return detected

    def _has_structure_graphics(self, pil_image: Image.Image) -> bool:
        """Check if a page image contains large localized graphic regions.

        Chemical structure pages have dark pixels clustered in localized regions,
        unlike text pages where dark pixels span nearly the full page width.

        Strategy:
        - Divide page into horizontal strips
        - For each strip, measure the "spread" of dark pixels (range / page width)
        - If spread < threshold, dark pixels are localized (likely structure graphics)
        - Count strips with localized clusters; if >= threshold, it's a structure page
        """
        gray = pil_image.convert('L')
        width, height = gray.size
        pixels = gray.load()

        structure_strips = 0

        for strip_top in range(0, height - self.STRIP_HEIGHT, self.STRIP_HEIGHT):
            strip_bottom = min(strip_top + self.STRIP_HEIGHT, height)

            # Collect x-positions of dark pixels in this strip
            dark_xs = []
            for y in range(strip_top, strip_bottom):
                for x in range(width):
                    if pixels[x, y] < self.DARK_THRESHOLD:
                        dark_xs.append(x)

            if not dark_xs:
                continue

            dark_ratio = len(dark_xs) / (width * (strip_bottom - strip_top))
            if dark_ratio < self.MIN_DARK_RATIO:
                continue

            # Measure spread: how wide are the dark pixels distributed?
            x_min = min(dark_xs)
            x_max = max(dark_xs)
            spread = (x_max - x_min) / width

            # Localized = dark pixels don't span full width
            if spread < self.MAX_SPREAD_RATIO:
                # Also check that the cluster is large enough (not just a small mark)
                cluster_width = x_max - x_min
                if cluster_width >= self.MIN_CLUSTER_SIZE:
                    structure_strips += 1

        return structure_strips >= self.MIN_STRUCTURE_STRIPS

    def _has_dense_structure_region(self, pil_image: Image.Image) -> bool:
        """Check if page has dense localized regions characteristic of chemical structures.

        Designed for rescue pass context: detects structures embedded in dense text
        that spread-based analysis can't distinguish from text columns.

        Uses cell-based density analysis — divides the page into a grid of small
        cells and looks for cells with high dark pixel density. Chemical structures
        have thick lines (bonds, rings) creating cells with >30% density, while text
        cells stay under ~28%. Also checks for vertical adjacency of moderately dense
        cells (>22%), indicating a 2D structure region rather than a 1D text line.

        Criterion: (vertically_clustered_cells >= 4) OR (max_cell_density > 0.35)
        """
        import numpy as np

        CELL_SIZE = 20
        DENSE_THRESHOLD = 0.22
        MAX_DENSITY_THRESHOLD = 0.35
        MIN_VERT_CLUSTERED = 4

        gray = pil_image.convert('L')
        arr = np.array(gray)
        height, width = arr.shape

        # Skip margins
        x_start = int(width * 0.05)
        x_end = int(width * 0.95)
        y_start = int(height * 0.08)
        y_end = int(height * 0.92)

        dark = (arr[y_start:y_end, x_start:x_end] < self.DARK_THRESHOLD).astype(np.float32)

        n_rows = (dark.shape[0] - CELL_SIZE) // CELL_SIZE + 1
        n_cols = (dark.shape[1] - CELL_SIZE) // CELL_SIZE + 1

        if n_rows < 2 or n_cols < 2:
            return False

        # Build density grid
        grid = np.zeros((n_rows, n_cols))
        for r in range(n_rows):
            for c in range(n_cols):
                cy = r * CELL_SIZE
                cx = c * CELL_SIZE
                grid[r, c] = dark[cy:cy+CELL_SIZE, cx:cx+CELL_SIZE].mean()

        # A single very dense cell indicates a structure (thick lines/bonds)
        if grid.max() > MAX_DENSITY_THRESHOLD:
            return True

        # Check for vertical adjacency of moderately dense cells —
        # structures span multiple cell rows (2D), text headers don't
        dense = grid > DENSE_THRESHOLD
        vert_clustered = 0
        for r in range(n_rows):
            for c in range(n_cols):
                if not dense[r, c]:
                    continue
                if (r > 0 and dense[r-1, c]) or (r < n_rows - 1 and dense[r+1, c]):
                    vert_clustered += 1

        return vert_clustered >= MIN_VERT_CLUSTERED

    def _has_bio_table_indicators(self, pil_image: Image.Image) -> bool:
        """Check if page has tabular layout (potential bio data table).

        Bio data tables have a regular grid pattern — evenly spaced columns
        of numbers with consistent vertical alignment.

        Strategy:
        - Project dark pixels onto the x-axis to find column peaks
        - Project dark pixels onto the y-axis to find row peaks
        - If both have regular spacing, it's likely a table
        """
        gray = pil_image.convert('L')
        width, height = gray.size
        pixels = gray.load()

        # Skip margins (top/bottom 10%, left/right 5%)
        x_start = int(width * 0.05)
        x_end = int(width * 0.95)
        y_start = int(height * 0.10)
        y_end = int(height * 0.90)

        # Build x-projection (count dark pixels per column)
        x_proj = [0] * width
        y_proj = [0] * height

        for y in range(y_start, y_end, 2):  # Sample every other row for speed
            for x in range(x_start, x_end):
                if pixels[x, y] < self.DARK_THRESHOLD:
                    x_proj[x] += 1
                    y_proj[y] += 1

        # Find column peaks: segments where x_proj is above average
        avg_x = sum(x_proj[x_start:x_end]) / max(1, x_end - x_start)
        if avg_x == 0:
            return False

        threshold_x = avg_x * 1.5
        in_peak = False
        col_peaks = 0
        gap_count = 0

        for x in range(x_start, x_end):
            if x_proj[x] > threshold_x:
                if not in_peak:
                    col_peaks += 1
                    in_peak = True
                gap_count = 0
            else:
                gap_count += 1
                if gap_count > width * 0.03:  # Gap threshold
                    in_peak = False

        # Find row peaks similarly
        avg_y = sum(y_proj[y_start:y_end]) / max(1, y_end - y_start)
        if avg_y == 0:
            return False

        threshold_y = avg_y * 0.8
        in_peak = False
        row_peaks = 0
        gap_count = 0

        for y in range(y_start, y_end):
            if y_proj[y] > threshold_y:
                if not in_peak:
                    row_peaks += 1
                    in_peak = True
                gap_count = 0
            else:
                gap_count += 1
                if gap_count > height * 0.02:
                    in_peak = False

        return col_peaks >= self.TABLE_GRID_MIN_COLS and row_peaks >= self.TABLE_GRID_MIN_ROWS
