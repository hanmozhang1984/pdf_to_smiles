"""LLM-based compound classification using Claude Haiku Vision.

Classifies detected chemical structures as 'example_compound' (specific compounds
from the patent's examples section) or 'other' (Markush/generic structures,
synthesis intermediates, reagents, claims section structures).

Uses full-page context: numbered red boxes are drawn around each detected structure,
and Claude sees the surrounding labels, table headers, and synthesis arrows to make
context-aware classification decisions.

When patent section bounds are available (from PatentSectionDetector / P2), the
classifier receives additional context about which section the page belongs to,
significantly improving accuracy.
"""

import base64
import io
import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

_MODEL = "claude-haiku-4-5-20251001"

_CLASSIFY_PROMPT = """\
You are classifying chemical structures on a pharmaceutical patent page.
I've drawn {n} red boxes around structures. Each box has a small red number label
(1, 2, 3, ...) at its top-left corner. Use THESE red box numbers as keys — NOT the
example/compound numbers printed on the page.

{section_context}
For each red-boxed structure, provide TWO things:
  (a) Its classification type (see rules below).
  (b) The compound/example number printed on the page near that structure — this is
      the patent's own number, NOT the red box number. Look for labels like
      "Example 25", "Ex. 7", "Compound 100-7", "Cpd. 12a", or a row number in a
      table. Return just the number/ID part (e.g., "25", "100-7", "12a").
      Return null if no compound number is visible near the structure (intermediates,
      Markush, reagents, etc. typically have no compound number).

Classify each structure by checking these rules IN ORDER:

1. INSIDE A TABLE with column headers like "Example", "Structure", "Name", "No."?
   -> "example_compound". Tables with example/compound numbers and structure
   drawings are ALWAYS example compounds, even if the page also has synthesis
   schemes elsewhere. Look for numbered rows (1, 2, 3... or 1a, 1b...).

2. Has an associated "Example X", "Ex. #", "Compound X", or "Cpd. X" label nearby?
   -> "example_compound" (the label may be separated by IUPAC names, molecular
   formulas, or synthesis text — look within ~2cm above or below the structure).

3. Is the FINAL product at the END of a multi-step synthesis scheme (last structure
   in a chain of arrows), AND is this page in the Examples section of the patent?
   -> "example_compound". The final product of an example synthesis IS the example
   compound. But intermediates in that same scheme are NOT — only the last product.

4. Everything else -> "other":
   - Markush/generic structures (R-groups like R¹, R², "Formula (I)", variable
     bonds shown as dashed lines, "wherein R is...")
   - Synthesis INTERMEDIATES — structures that appear BEFORE reaction arrows, or
     labeled with codes like "Int-X", "Intermediate", step numbers (Step 1, Step 2),
     or letter-number codes (A-1, G-2, C43, P1, SM-1)
   - Starting materials and reagents
   - Reference compounds or known drugs shown for comparison
   - Fragments with wavy bonds (partial structures)
   - Structures in patent Claims section (even if fully defined with no R-groups)
   - Structures labeled as "Formula (I)", "Formula (II)" etc. (general formulas)

IMPORTANT: A single page can have BOTH example compounds (in a table at the top)
AND synthesis intermediates (in a scheme below). Classify each structure individually.

Respond ONLY with JSON: {{"1": {{"type": "example_compound", "id": "25"}}, "2": {{"type": "other", "id": null}}, ...}}"""

# Maximum image dimension for Claude Vision optimal performance
_MAX_IMAGE_DIM = 1568


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


def _resize_for_api(image: Image.Image) -> Image.Image:
    """Resize image if longer edge exceeds Claude Vision's optimal limit."""
    w, h = image.size
    if max(w, h) <= _MAX_IMAGE_DIM:
        return image
    scale = _MAX_IMAGE_DIM / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return image.resize((new_w, new_h), Image.LANCZOS)


class LLMCompoundClassifier:
    """Classify detected structures as example_compound or other using Claude Haiku."""

    def __init__(self):
        self._client = None

    def _get_client(self):
        """Lazy-initialize the Anthropic client."""
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    def classify_page_structures(
        self,
        page_image: Image.Image,
        structure_boxes: List[Tuple[int, int, int, int]],
        page_num: Optional[int] = None,
        section_bounds=None,
    ) -> List[Dict]:
        """Classify each structure on a page and extract compound IDs.

        Draws numbered red boxes on the page image at each structure location,
        sends the annotated image to Claude Haiku, and returns classifications
        along with any compound/example numbers visible on the page.

        Args:
            page_image: Full page image (PIL Image).
            structure_boxes: List of (x1, y1, x2, y2) bounding boxes for each
                detected structure, in page-image pixel coordinates.
            page_num: 1-indexed page number (for section context).
            section_bounds: SectionBounds from PatentSectionDetector (optional).

        Returns:
            List of dicts with keys "type" (str) and "id" (Optional[str]),
            one per structure. Falls back to {"type": "other", "id": None}
            on any error.
        """
        n = len(structure_boxes)
        if n == 0:
            return []

        # Draw numbered red boxes on a copy of the page image
        annotated = page_image.copy().convert("RGB")
        draw = ImageDraw.Draw(annotated)

        # Try to use a reasonably-sized font for the numbers
        font = None
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        except (OSError, IOError):
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            except (OSError, IOError):
                font = ImageFont.load_default()

        for idx, (x1, y1, x2, y2) in enumerate(structure_boxes):
            label = str(idx + 1)
            # Draw red rectangle (3px wide)
            for offset in range(3):
                draw.rectangle(
                    [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                    outline="red",
                )
            # Draw number label above the box
            text_x = x1
            text_y = max(0, y1 - 24)
            # White background for readability
            bbox = draw.textbbox((text_x, text_y), label, font=font)
            draw.rectangle(
                [bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2],
                fill="white",
            )
            draw.text((text_x, text_y), label, fill="red", font=font)

        # Resize for API
        annotated = _resize_for_api(annotated)
        image_data = _encode_image(annotated)

        # Build section context string
        section_context = self._build_section_context(page_num, section_bounds)

        # Build prompt
        prompt = _CLASSIFY_PROMPT.format(n=n, section_context=section_context)

        # Call Claude Haiku
        client = self._get_client()
        response = client.messages.create(
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
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
        )

        raw_text = response.content[0].text.strip()
        logger.debug("Page %s raw LLM response: %s", page_num or "?", raw_text)

        results = self._parse_response(response, n)
        logger.debug(
            "Page %s classification: %s",
            page_num or "?",
            {str(i + 1): r for i, r in enumerate(results)},
        )
        return results

    @staticmethod
    def _build_section_context(
        page_num: Optional[int],
        section_bounds,
    ) -> str:
        """Build a context string describing which patent section this page is in.

        This significantly improves classification accuracy by giving Claude prior
        knowledge about the page's location in the patent structure.
        """
        if section_bounds is None or not section_bounds.is_valid:
            if page_num is not None:
                return f"This is page {page_num} of the patent.\n"
            return ""

        parts = [f"This is page {page_num} of {section_bounds.total_pages}."]

        examples_start = section_bounds.examples_start
        examples_end = section_bounds.examples_end

        if page_num is not None and examples_start is not None:
            if page_num < examples_start:
                parts.append(
                    f"This page is BEFORE the Examples section (Examples start at "
                    f"page {examples_start}). Structures here are likely from the "
                    f"Description section — expect Markush/generic structures, "
                    f"general formulas, or illustrative schemes. Classify as \"other\" "
                    f"unless you see clear \"Example X\" labels."
                )
            elif section_bounds.claims_start and page_num >= section_bounds.claims_start:
                parts.append(
                    f"This page is in the CLAIMS section (Claims start at page "
                    f"{section_bounds.claims_start}). ALL structures on Claims pages "
                    f"should be classified as \"other\", even if they look like "
                    f"specific compounds."
                )
            else:
                parts.append(
                    f"This page is in the EXAMPLES section (pages "
                    f"{examples_start}-{examples_end}). Structures here are likely "
                    f"example compounds, but still check for synthesis intermediates "
                    f"and Markush/generic structures."
                )

        return " ".join(parts) + "\n"

    @staticmethod
    def _parse_entry(value) -> Dict:
        """Parse a single response entry into {"type": str, "id": Optional[str]}.

        Handles both new format ({"type": "example_compound", "id": "25"})
        and old flat format ("example_compound").
        """
        if isinstance(value, dict):
            entry_type = value.get("type", "other")
            if entry_type not in ("example_compound", "other"):
                entry_type = "other"
            entry_id = value.get("id")
            # Normalize id: convert non-string/empty to None
            if entry_id is not None:
                entry_id = str(entry_id).strip()
                if not entry_id or entry_id.lower() == "null":
                    entry_id = None
            return {"type": entry_type, "id": entry_id}
        elif isinstance(value, str) and value in ("example_compound", "other"):
            # Old flat format — backward compatible
            return {"type": value, "id": None}
        return None

    def _parse_response(self, response, n: int) -> List[Dict]:
        """Parse Claude's JSON response into a list of classification dicts."""
        text = response.content[0].text.strip()

        # Extract JSON object from response — handle code blocks, trailing text, etc.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse compound classifier response: %s", text)
            return [{"type": "other", "id": None}] * n

        # Try direct box-number keys first ("1", "2", ...)
        results = []
        for i in range(1, n + 1):
            value = data.get(str(i))
            entry = self._parse_entry(value)
            results.append(entry)

        # If some keys are missing, the model may have used the patent's Example
        # numbers instead of our red-box numbers. Fall back to values in key order.
        if any(r is None for r in results) and len(data) == n:
            def _numeric_sort_key(kv):
                """Sort keys numerically, falling back to string sort."""
                digits = re.sub(r'[^0-9]', '', kv[0])
                return int(digits) if digits else float('inf')

            sorted_entries = []
            for _, v in sorted(data.items(), key=_numeric_sort_key):
                entry = self._parse_entry(v)
                sorted_entries.append(entry if entry else {"type": "other", "id": None})
            results = sorted_entries

        # Fill any remaining Nones with default
        results = [r if r is not None else {"type": "other", "id": None} for r in results]

        return results
