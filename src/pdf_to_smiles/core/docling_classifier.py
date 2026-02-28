"""Page classifier using Docling (Apache 2.0) for detecting pages with chemical structures or bio data.

Uses IBM's Docling library with the DocLayNet RT-DETR model for document layout analysis.
Replaces DocLayout-YOLO (AGPL-3.0) to avoid copyleft licensing constraints.

Docling processes entire PDFs at once, so detect_structure_pages() is the primary entry point.
classify_page() is provided for interface compatibility but is less efficient (writes a temp PDF).
"""

import logging
import tempfile
import os
from typing import Dict, List, Optional

from PIL import Image

from .doclayout_classifier import PageClassification

logger = logging.getLogger(__name__)

# DocLayNet labels → our classification groups
_STRUCTURE_LABELS = {"picture", "formula", "chart"}
_TABLE_LABELS = {"table"}
_TEXT_LABELS = {
    "caption", "text", "title", "section_header", "list_item",
    "footnote", "page_footer", "page_header", "code", "paragraph",
    "reference", "document_index",
}

CONFIDENCE_THRESHOLD = 0.25


class DoclingClassifier:
    """Classify PDF pages using Docling document layout analysis (Apache 2.0)."""

    DPI = 200  # Match rendering resolution for consistency

    def __init__(self):
        self._converter = None

    def _load_model(self):
        """Lazy-init the Docling converter with layout-only settings."""
        if self._converter is not None:
            return

        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions

        pipeline_options = PdfPipelineOptions(
            do_ocr=False,
            do_table_structure=False,
            do_formula_enrichment=False,
            do_code_enrichment=False,
            do_picture_classification=False,
            do_picture_description=False,
        )

        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            }
        )
        logger.info("Docling converter initialized.")

    def classify_pdf(self, pdf_path: str) -> Dict[int, PageClassification]:
        """Classify all pages in a PDF at once.

        This is the efficient entry point — Docling processes entire PDFs natively.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Dict mapping 1-indexed page number to PageClassification.
        """
        self._load_model()

        result = self._converter.convert(pdf_path)

        classifications = {}
        for page in result.pages:
            page_no = page.page_no  # 1-indexed
            layout = page.predictions.layout

            if layout is None:
                classifications[page_no] = PageClassification()
                continue

            categories = {}
            confidence_scores = {}

            for cluster in layout.clusters:
                label = cluster.label.value
                conf = cluster.confidence

                if conf < CONFIDENCE_THRESHOLD:
                    continue

                categories[label] = categories.get(label, 0) + 1
                confidence_scores[label] = max(
                    confidence_scores.get(label, 0.0), conf
                )

            has_structures = any(c in categories for c in _STRUCTURE_LABELS)
            has_tables = any(c in categories for c in _TABLE_LABELS)

            classifications[page_no] = PageClassification(
                has_structures=has_structures,
                has_tables=has_tables,
                is_text_only=not has_structures and not has_tables,
                categories=categories,
                confidence_scores=confidence_scores,
            )

        return classifications

    def classify_page(self, pil_image: Image.Image) -> PageClassification:
        """Classify a single page image.

        Note: Less efficient than classify_pdf() since Docling works on full PDFs.
        This writes a temporary single-page PDF for compatibility with the interface
        used by HybridClassifier.

        Args:
            pil_image: PIL Image of the page.

        Returns:
            PageClassification with detected categories.
        """
        # Convert PIL image to a single-page PDF for Docling
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
            # Save as PDF (Pillow supports this natively for RGB images)
            rgb_image = pil_image.convert("RGB")
            rgb_image.save(tmp_path, format="PDF")

        try:
            results = self.classify_pdf(tmp_path)
            return results.get(1, PageClassification())
        finally:
            os.unlink(tmp_path)

    def detect_structure_pages(
        self,
        pdf_path: str,
        progress_callback=None,
    ) -> List[int]:
        """Scan PDF and return 1-indexed page numbers containing structures or bio data.

        Same interface as PageClassifier/DocLayoutClassifier.

        After Docling classification, runs a heuristic rescue pass on pages marked
        text_only to catch structures embedded in dense text that Docling misses.

        Args:
            pdf_path: Path to the PDF file.
            progress_callback: Optional callable(current_page, total_pages).

        Returns:
            Sorted list of 1-indexed page numbers.
        """
        import pypdfium2 as pdfium
        from .ppdoclayout_rescue import PPDocLayoutRescue

        # Get total page count for progress reporting
        doc = pdfium.PdfDocument(pdf_path)
        total_pages = len(doc)
        doc.close()

        if progress_callback:
            progress_callback(0, total_pages)

        # Docling processes the entire PDF at once
        classifications = self.classify_pdf(pdf_path)

        detected = set()
        text_only_pages = []
        for page_no in sorted(classifications.keys()):
            if progress_callback:
                progress_callback(page_no, total_pages)
            if classifications[page_no].should_process:
                detected.add(page_no)
            elif classifications[page_no].is_text_only:
                text_only_pages.append(page_no)

        # Rescue pass: run PP-DocLayout-L on text_only pages (in subprocess)
        rescue = PPDocLayoutRescue()
        rescued = rescue.rescue_pages(pdf_path, text_only_pages, dpi=self.DPI)
        detected.update(rescued)

        logger.info(
            "Docling rescue pass (PP-DocLayout): %d text_only pages checked, %d rescued: %s",
            len(text_only_pages),
            len(rescued),
            rescued,
        )

        if progress_callback:
            progress_callback(total_pages, total_pages)

        return sorted(detected)
