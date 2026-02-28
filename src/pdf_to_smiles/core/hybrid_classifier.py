"""Hybrid page classifier combining Docling with Claude Vision verification.

Uses Docling as the fast primary classifier and only calls Claude Vision API
for pages where Docling produces low-confidence or no-detection results.
This catches the ~5% of pages where Docling is wrong at minimal API cost.
"""

import logging
from typing import List

import pypdfium2 as pdfium
from PIL import Image

from .docling_classifier import DoclingClassifier
from .doclayout_classifier import PageClassification
from .llm_layout_analyzer import LLMLayoutAnalyzer
from .ppdoclayout_rescue import PPDocLayoutRescue

logger = logging.getLogger(__name__)


class HybridClassifier:
    """Docling primary + Claude Vision verification on ambiguous pages."""

    def __init__(self, confidence_threshold: float = 0.60):
        self._primary = DoclingClassifier()
        self._llm = LLMLayoutAnalyzer()
        self._confidence_threshold = confidence_threshold
        self._stats = {"total": 0, "primary_only": 0, "llm_verified": 0, "llm_overridden": 0}

    @property
    def stats(self):
        """Return classification statistics for diagnostics."""
        return dict(self._stats)

    def classify_page(self, pil_image: Image.Image) -> PageClassification:
        """Classify with Docling first, verify with Claude if ambiguous.

        Decision logic:
        1. Run Docling layout analysis
        2. If max confidence > threshold: trust Docling (no API call)
        3. If max confidence <= threshold or no detections: verify with Claude Vision

        Args:
            pil_image: PIL Image of the page (should be ~200 DPI).

        Returns:
            PageClassification with the best available result.
        """
        self._stats["total"] += 1

        # Primary classification with Docling
        primary_result = self._primary.classify_page(pil_image)

        # Check max confidence across all detected categories
        max_confidence = max(primary_result.confidence_scores.values()) if primary_result.confidence_scores else 0.0

        if max_confidence > self._confidence_threshold:
            # High confidence: trust Docling
            self._stats["primary_only"] += 1
            logger.debug(
                "Docling high confidence (%.2f): %s",
                max_confidence,
                primary_result.categories,
            )
            return primary_result

        # Low confidence or no detections: verify with Claude Vision
        logger.info(
            "Docling low confidence (%.2f), verifying with Claude Vision. Categories: %s",
            max_confidence,
            primary_result.categories,
        )
        self._stats["llm_verified"] += 1

        try:
            llm_result = self._llm.classify_page(pil_image)
        except Exception as e:
            logger.warning("Claude Vision verification failed: %s. Using Docling result.", e)
            return primary_result

        # If Claude disagrees with Docling, use Claude's result
        if llm_result.should_process != primary_result.should_process:
            self._stats["llm_overridden"] += 1
            logger.info(
                "Claude Vision overrides Docling: Docling=%s, Claude=%s",
                "process" if primary_result.should_process else "skip",
                "process" if llm_result.should_process else "skip",
            )
            return llm_result

        return primary_result

    def detect_structure_pages(
        self,
        pdf_path: str,
        progress_callback=None,
    ) -> List[int]:
        """Scan PDF and return 1-indexed page numbers containing structures or bio data.

        Optimized for Docling: runs layout analysis on the full PDF first,
        then only renders and sends to Claude Vision the low-confidence pages.

        Args:
            pdf_path: Path to the PDF file.
            progress_callback: Optional callable(current_page, total_pages).

        Returns:
            Sorted list of 1-indexed page numbers.
        """
        # Step 1: Run Docling on the entire PDF at once
        doc = pdfium.PdfDocument(pdf_path)
        total_pages = len(doc)

        if progress_callback:
            progress_callback(0, total_pages)

        primary_results = self._primary.classify_pdf(pdf_path)

        # Step 2: For each page, trust Docling if high confidence, else verify with Claude
        detected = []

        for page_idx in range(total_pages):
            page_no = page_idx + 1
            if progress_callback:
                progress_callback(page_no, total_pages)

            self._stats["total"] += 1
            result = primary_results.get(page_no, PageClassification())
            max_confidence = max(result.confidence_scores.values()) if result.confidence_scores else 0.0

            if max_confidence > self._confidence_threshold:
                # High confidence: trust Docling
                self._stats["primary_only"] += 1
                if result.should_process:
                    detected.append(page_no)
                continue

            # Low confidence: render page and verify with Claude Vision
            logger.info(
                "Page %d: Docling low confidence (%.2f), verifying with Claude Vision.",
                page_no, max_confidence,
            )
            self._stats["llm_verified"] += 1

            try:
                page = doc[page_idx]
                bitmap = page.render(scale=self._primary.DPI / 72)
                pil_image = bitmap.to_pil()
                llm_result = self._llm.classify_page(pil_image)
            except Exception as e:
                logger.warning("Claude Vision failed for page %d: %s. Using Docling result.", page_no, e)
                if result.should_process:
                    detected.append(page_no)
                continue

            if llm_result.should_process != result.should_process:
                self._stats["llm_overridden"] += 1
                logger.info(
                    "Page %d: Claude overrides Docling: Docling=%s, Claude=%s",
                    page_no,
                    "process" if result.should_process else "skip",
                    "process" if llm_result.should_process else "skip",
                )
                if llm_result.should_process:
                    detected.append(page_no)
            elif result.should_process:
                detected.append(page_no)

        # Rescue pass: run pixel heuristic on high-confidence text_only pages
        # that skipped Claude Vision verification
        detected_set = set(detected)
        text_only_pages = []
        for page_idx in range(total_pages):
            page_no = page_idx + 1
            if page_no in detected_set:
                continue
            result = primary_results.get(page_no, PageClassification())
            max_conf = max(result.confidence_scores.values()) if result.confidence_scores else 0.0
            # Only rescue high-confidence text_only pages (low-confidence ones already went through Claude Vision)
            if result.is_text_only and max_conf > self._confidence_threshold:
                text_only_pages.append(page_no)

        rescue = PPDocLayoutRescue()
        rescued = rescue.rescue_pages(pdf_path, text_only_pages, dpi=self._primary.DPI)
        detected.extend(rescued)

        doc.close()

        self._stats["rescued"] = len(rescued)
        logger.info(
            "Hybrid classifier stats: %d pages, %d primary-only, %d LLM-verified, "
            "%d LLM-overridden, %d rescued by PP-DocLayout",
            self._stats["total"],
            self._stats["primary_only"],
            self._stats["llm_verified"],
            self._stats["llm_overridden"],
            len(rescued),
        )
        if rescued:
            logger.info("Rescued pages: %s", rescued)

        return sorted(detected)
