"""Hybrid page classifier combining DocLayout-YOLO with Claude Vision verification.

Uses YOLO as the fast primary classifier and only calls Claude Vision API
for pages where YOLO produces low-confidence or no-detection results.
This catches the ~5% of pages where YOLO is wrong at minimal API cost.
"""

import logging
from typing import List

import pypdfium2 as pdfium
from PIL import Image

from .doclayout_classifier import DocLayoutClassifier, PageClassification
from .llm_layout_analyzer import LLMLayoutAnalyzer

logger = logging.getLogger(__name__)


class HybridClassifier:
    """DocLayout-YOLO primary + Claude Vision verification on ambiguous pages."""

    def __init__(self, yolo_confidence_threshold: float = 0.60):
        self._yolo = DocLayoutClassifier()
        self._llm = LLMLayoutAnalyzer()
        self._confidence_threshold = yolo_confidence_threshold
        self._stats = {"total": 0, "yolo_only": 0, "llm_verified": 0, "llm_overridden": 0}

    @property
    def stats(self):
        """Return classification statistics for diagnostics."""
        return dict(self._stats)

    def classify_page(self, pil_image: Image.Image) -> PageClassification:
        """Classify with YOLO first, verify with Claude if ambiguous.

        Decision logic:
        1. Run DocLayout-YOLO
        2. If max confidence > threshold: trust YOLO (no API call)
        3. If max confidence <= threshold or no detections: verify with Claude Vision

        Args:
            pil_image: PIL Image of the page (should be ~200 DPI).

        Returns:
            PageClassification with the best available result.
        """
        self._stats["total"] += 1

        # Primary classification with YOLO
        yolo_result = self._yolo.classify_page(pil_image)

        # Check max confidence across all detected categories
        max_confidence = max(yolo_result.confidence_scores.values()) if yolo_result.confidence_scores else 0.0

        if max_confidence > self._confidence_threshold:
            # High confidence: trust YOLO
            self._stats["yolo_only"] += 1
            logger.debug(
                "YOLO high confidence (%.2f): %s",
                max_confidence,
                yolo_result.categories,
            )
            return yolo_result

        # Low confidence or no detections: verify with Claude Vision
        logger.info(
            "YOLO low confidence (%.2f), verifying with Claude Vision. Categories: %s",
            max_confidence,
            yolo_result.categories,
        )
        self._stats["llm_verified"] += 1

        try:
            llm_result = self._llm.classify_page(pil_image)
        except Exception as e:
            logger.warning("Claude Vision verification failed: %s. Using YOLO result.", e)
            return yolo_result

        # If Claude disagrees with YOLO, use Claude's result
        if llm_result.should_process != yolo_result.should_process:
            self._stats["llm_overridden"] += 1
            logger.info(
                "Claude Vision overrides YOLO: YOLO=%s, Claude=%s",
                "process" if yolo_result.should_process else "skip",
                "process" if llm_result.should_process else "skip",
            )
            return llm_result

        return yolo_result

    def detect_structure_pages(
        self,
        pdf_path: str,
        progress_callback=None,
    ) -> List[int]:
        """Scan PDF and return 1-indexed page numbers containing structures or bio data.

        Same interface as PageClassifier/DocLayoutClassifier.

        Args:
            pdf_path: Path to the PDF file.
            progress_callback: Optional callable(current_page, total_pages).

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
            bitmap = page.render(scale=self._yolo.DPI / 72)
            pil_image = bitmap.to_pil()

            classification = self.classify_page(pil_image)
            if classification.should_process:
                detected.append(page_idx + 1)

        doc.close()

        logger.info(
            "Hybrid classifier stats: %d pages, %d YOLO-only, %d LLM-verified, %d LLM-overridden",
            self._stats["total"],
            self._stats["yolo_only"],
            self._stats["llm_verified"],
            self._stats["llm_overridden"],
        )

        return detected
