"""Claude Vision page classifier for verifying document layout analysis.

Uses Claude Haiku Vision to classify PDF pages when DocLayout-YOLO produces
low-confidence results. Acts as a verification layer, not a replacement.
"""

import base64
import io
import json
import logging
import os
from typing import List, Optional

import pypdfium2 as pdfium
from PIL import Image

from .doclayout_classifier import PageClassification

logger = logging.getLogger(__name__)

_CLASSIFY_PROMPT = """\
Analyze this PDF page image. Determine what content it contains.

Respond with JSON only, no other text:
{
  "has_chemical_structures": true/false,
  "has_bio_data_tables": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}

Definitions:
- has_chemical_structures: Page contains chemical structure diagrams (molecular drawings with bonds, rings, atoms). NOT reaction arrows alone, NOT just chemical names/formulas in text.
- has_bio_data_tables: Page contains data tables with biological assay results (IC50, EC50, Ki, % inhibition, etc.) with numeric data in columns.

Be precise: reaction scheme overview pages without individual extractable structures should be false for has_chemical_structures."""

_MODEL = "claude-haiku-4-5-20251001"


def is_available() -> bool:
    """Check if anthropic SDK is installed and ANTHROPIC_API_KEY is set."""
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _encode_image(pil_image: Image.Image) -> str:
    """Base64-encode a PIL image as PNG."""
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def classify_page(image: Image.Image) -> PageClassification:
    """Classify a page image using Claude Vision.

    Args:
        image: PIL Image of the page.

    Returns:
        PageClassification with results from Claude Vision.
    """
    analyzer = LLMLayoutAnalyzer()
    return analyzer.classify_page(image)


class LLMLayoutAnalyzer:
    """Claude Vision page classifier -- used as verification for DocLayout-YOLO."""

    CONFIDENCE_THRESHOLD = 0.60  # YOLO confidence below this triggers verification
    DPI = 200  # Match YOLO rendering resolution

    def __init__(self):
        self._client = None

    def _get_client(self):
        """Lazy-initialize the Anthropic client."""
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    def classify_page(self, pil_image: Image.Image) -> PageClassification:
        """Send page image to Claude Haiku Vision, return PageClassification.

        Args:
            pil_image: PIL Image of the page.

        Returns:
            PageClassification with Claude Vision's assessment.
        """
        image_data = _encode_image(pil_image)
        client = self._get_client()

        response = client.messages.create(
            model=_MODEL,
            max_tokens=256,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": _CLASSIFY_PROMPT,
                        },
                    ],
                }
            ],
        )

        return self._parse_response(response)

    def _parse_response(self, response) -> PageClassification:
        """Parse Claude's JSON response into a PageClassification."""
        text = response.content[0].text.strip()

        # Extract JSON from response (handle markdown code blocks)
        if text.startswith("```"):
            # Strip ```json ... ```
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse Claude Vision response: %s", text)
            # Conservative fallback: don't process
            return PageClassification(
                has_structures=False,
                has_tables=False,
                is_text_only=True,
                categories={"llm_parse_error": 1},
                confidence_scores={},
            )

        has_structures = data.get("has_chemical_structures", False)
        has_tables = data.get("has_bio_data_tables", False)
        confidence = data.get("confidence", 0.5)
        reasoning = data.get("reasoning", "")

        categories = {}
        confidence_scores = {}
        if has_structures:
            categories["figure"] = 1
            confidence_scores["figure"] = confidence
        if has_tables:
            categories["table"] = 1
            confidence_scores["table"] = confidence

        is_text_only = not has_structures and not has_tables

        if reasoning:
            logger.debug("Claude Vision reasoning: %s", reasoning)

        return PageClassification(
            has_structures=has_structures,
            has_tables=has_tables,
            is_text_only=is_text_only,
            categories=categories,
            confidence_scores=confidence_scores,
        )

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
            bitmap = page.render(scale=self.DPI / 72)
            pil_image = bitmap.to_pil()

            classification = self.classify_page(pil_image)
            if classification.should_process:
                detected.append(page_idx + 1)

        doc.close()
        return detected
