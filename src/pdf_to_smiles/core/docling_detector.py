"""Structure detector using cached Docling layout analysis.

Wraps DoclingClassifier to provide structure detection with bounding boxes,
reusing the RT-DETR layout clusters that are already computed during page
scanning. This avoids the heuristic OpenCV pipeline (LightweightDetector)
which can truncate/miscrop structures on complex synthesis pages.
"""

import logging
from typing import List, Optional, Tuple

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

    def __init__(self, docling_classifier):
        """Initialize with a DoclingClassifier that has cached layout data.

        Args:
            docling_classifier: A DoclingClassifier instance that has already
                processed the PDF (layout clusters are cached).
        """
        self._classifier = docling_classifier

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
