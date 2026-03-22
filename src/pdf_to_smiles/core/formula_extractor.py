"""Extract molecular formulas and MS data from patent page images using Claude Vision.

Uses Claude Haiku Vision to OCR patent pages and extract analytical data
(molecular formulas, MS/HRMS/LCMS masses) for validating OCSR output.
"""

import base64
import io
import json
import logging
from dataclasses import dataclass
from typing import List, Optional

from PIL import Image

logger = logging.getLogger(__name__)

_MODEL = "claude-haiku-4-5-20251001"

_EXTRACTION_PROMPT = """\
Extract molecular formula and mass spectrometry data from this patent page.
For each compound with analytical data, return a JSON object with these fields:
- "compound_id": the example/compound number (e.g. "85", "1a"), or null if unclear
- "formula": the molecular formula exactly as stated (e.g. "C28H30F4N8O2S")
- "mass": the observed or calculated m/z value as a number (e.g. 619.2), or null
- "adduct": the adduct type (e.g. "[M+H]+", "[M+Na]+"), or null
- "source": one of "ms", "hrms", "lcms", "formula_only"

Return a JSON array. If no MS/HRMS/LCMS/analytical data on this page, return [].
Only extract data explicitly stated on the page — do not compute or infer values.
"""


@dataclass
class FormulaReference:
    """Reference formula and mass data extracted from patent analytical sections."""

    molecular_formula: str  # e.g., "C28H30F4N8O2S"
    expected_mh_mass: Optional[float]  # [M+H]+ or [M+Na]+ mass
    compound_id: Optional[str]  # e.g. "85" if detectable
    confidence: float  # 0.0-1.0
    source: str  # "ms", "hrms", "lcms", "formula_only"


class FormulaExtractor:
    """Extracts molecular formulas and MS data from patent pages via Claude Vision."""

    def __init__(self, api_key: str):
        """Initialize with Anthropic API key.

        Args:
            api_key: Anthropic API key for Claude Haiku Vision calls.
        """
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)

    def extract_from_page(
        self, page_image: Image.Image, page_num: int
    ) -> List[FormulaReference]:
        """Send page image to Claude Haiku with targeted prompt.

        Args:
            page_image: PIL Image of the rendered patent page.
            page_num: 1-indexed page number (for logging).

        Returns:
            List of FormulaReference objects extracted from the page.
            Returns empty list if no analytical data found.
        """
        # Resize large images for cost efficiency (max 1568px on longest side)
        img = _resize_for_api(page_image)

        # Encode image as base64 PNG
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        image_data = base64.b64encode(buf.getvalue()).decode("utf-8")

        try:
            response = self._client.messages.create(
                model=_MODEL,
                max_tokens=2048,
                temperature=0,
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
                            {"type": "text", "text": _EXTRACTION_PROMPT},
                        ],
                    }
                ],
            )
        except Exception as e:
            logger.warning("Claude Vision call failed for page %d: %s", page_num, e)
            return []

        raw_text = response.content[0].text.strip()
        return _parse_response(raw_text, page_num)

    def extract_batch(
        self, pages: List[tuple]
    ) -> dict:
        """Extract formulas from multiple pages.

        Args:
            pages: List of (page_image, page_num) tuples.

        Returns:
            Dict mapping page_num to list of FormulaReference objects.
        """
        result = {}
        for page_image, page_num in pages:
            refs = self.extract_from_page(page_image, page_num)
            result[page_num] = refs
        return result


def _resize_for_api(img: Image.Image, max_side: int = 1568) -> Image.Image:
    """Resize image so longest side is at most max_side pixels."""
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return img.resize((new_w, new_h), Image.LANCZOS)


def _parse_response(raw_text: str, page_num: int) -> List[FormulaReference]:
    """Parse Claude's JSON response into FormulaReference objects."""
    # Strip markdown code fences if present
    text = raw_text.strip()
    if text.startswith("```"):
        # Remove first line (```json or ```) and last line (```)
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3].rstrip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.debug("Failed to parse formula extraction JSON for page %d: %s", page_num, text[:200])
        return []

    if not isinstance(data, list):
        logger.debug("Formula extraction returned non-list for page %d", page_num)
        return []

    refs = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        formula = entry.get("formula")
        if formula is not None and not isinstance(formula, str):
            formula = None

        mass = entry.get("mass")
        if mass is not None:
            try:
                mass = float(mass)
            except (ValueError, TypeError):
                mass = None

        # Need at least a formula or a mass
        if not formula and mass is None:
            continue

        compound_id = entry.get("compound_id")
        if compound_id is not None:
            compound_id = str(compound_id).strip()

        source = entry.get("source", "formula_only")
        if source not in ("ms", "hrms", "lcms", "formula_only"):
            source = "formula_only"

        # Assign confidence based on source type
        confidence = {"hrms": 0.95, "lcms": 0.9, "ms": 0.85, "formula_only": 0.7}.get(
            source, 0.7
        )

        refs.append(
            FormulaReference(
                molecular_formula=formula or "",
                expected_mh_mass=mass,
                compound_id=compound_id,
                confidence=confidence,
                source=source,
            )
        )

    logger.debug("Page %d: extracted %d formula references", page_num, len(refs))
    return refs
