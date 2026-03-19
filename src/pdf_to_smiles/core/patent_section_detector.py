"""Detect patent section boundaries (Examples, Claims) to limit processing scope.

Chemistry patents follow a predictable structure:
Abstract -> Background -> Detailed Description -> Examples -> Bio Data -> Claims

By detecting where Examples starts and Claims begins, we can skip 30-50% of pages
that contain Markush structures (Description) or claim text (Claims).

Three-tier detection:
  Tier 1: pdfplumber text extraction (fast, free, works on text-based PDFs)
  Tier 2: pytesseract OCR on sampled pages (slower, works on image-based PDFs)
  Tier 3: Claude Vision on sampled pages (most robust, works on any PDF)
"""

import base64
import io
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from PIL import Image

logger = logging.getLogger(__name__)

# Try to import pytesseract for OCR fallback
try:
    import pytesseract
    from ..utils.paths import configure_tesseract
    HAS_TESSERACT = configure_tesseract()
except ImportError:
    HAS_TESSERACT = False


def _has_claude_vision() -> bool:
    """Check if Claude Vision is available (anthropic SDK + API key)."""
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


@dataclass
class SectionBounds:
    """Page boundaries for patent sections."""

    examples_start: Optional[int] = None  # 1-indexed page number
    claims_start: Optional[int] = None    # 1-indexed page number
    total_pages: int = 0
    detection_method: str = "none"        # "text", "ocr", "vision", or "none"

    @property
    def is_valid(self) -> bool:
        """Whether we found at least the Examples section start."""
        return self.examples_start is not None

    @property
    def examples_end(self) -> int:
        """Last page of the Examples section (inclusive, 1-indexed)."""
        if self.claims_start is not None:
            return self.claims_start - 1
        return self.total_pages

    def get_page_range(self) -> set:
        """Return set of 1-indexed page numbers covering the Examples section."""
        if not self.is_valid:
            return set()
        return set(range(self.examples_start, self.examples_end + 1))

    def summary(self) -> str:
        """Human-readable summary of detected bounds."""
        if not self.is_valid:
            return "No patent sections detected"
        parts = [f"Examples section: pages {self.examples_start}-{self.examples_end}"]
        if self.claims_start is not None:
            parts.append(f"Claims at page {self.claims_start}")
        parts.append(f"({self.detection_method} detection)")
        return ", ".join(parts)


# --- Section header patterns ---

# Examples section headers (marks start of useful content)
_EXAMPLES_PATTERNS = [
    # Standalone section headers (must be near start of line)
    re.compile(r'^\s*EXAMPLES?\s*AND\s+PREPARATIONS?\s*$', re.MULTILINE),
    re.compile(r'^\s*EXAMPLES?\s*$', re.MULTILINE),
    re.compile(r'^\s*EXPERIMENTAL\s+SECTION\s*$', re.MULTILINE),
    re.compile(r'^\s*EXPERIMENTAL\s*$', re.MULTILINE),
    # Compound headers: "SPECIFIC EXAMPLES", "SYNTHESIS OF EXAMPLES", etc.
    re.compile(r'^\s*(?:SPECIFIC|SYNTHESIS\s+OF|PREPARATIVE)\s+EXAMPLES?\s*$', re.MULTILINE),
    # "Non-Limiting Exemplary Compounds", "Exemplary Compounds", etc.
    re.compile(
        r'^\s*(?:NON[- ]LIMITING\s+)?EXEMPLARY\s+COMPOUNDS?\s*$',
        re.MULTILINE | re.IGNORECASE,
    ),
    # "Example 1:" or "Example 1." as a heading (not inline reference)
    # Must appear at/near start of line, not buried in a paragraph
    re.compile(r'^\s*Example\s+1\s*[:.]\s*', re.MULTILINE),
    # "EXAMPLE 1" in uppercase
    re.compile(r'^\s*EXAMPLE\s+1\s*[:.]*\s*', re.MULTILINE),
    # Flexible "Example 1" with prefix: "Synthesis of Example 1",
    # "Alternate Synthesis of Example 1", "Preparation of Example 1"
    re.compile(
        r'^\s*(?:(?:Alternate\s+)?Synthesis|Preparation)\s+of\s+Example\s+1\b',
        re.MULTILINE | re.IGNORECASE,
    ),
    # Two-column PDF fallback: pdfplumber merges columns so "EXAMPLES" may be
    # followed by text from the adjacent column (e.g., "EXAMPLES 50 yl ]-...")
    # Safe because _is_heading_context trusts all-caps matches.
    re.compile(r'^EXAMPLES?\b', re.MULTILINE),
]

# Claims section headers (marks end of useful content)
_CLAIMS_PATTERNS = [
    re.compile(r'^\s*WHAT\s+IS\s+CLAIMED\s+IS\s*:', re.MULTILINE),
    re.compile(r'^\s*CLAIMS?\s*$', re.MULTILINE),
    re.compile(r'^\s*THE\s+CLAIMS?\s*$', re.MULTILINE),
    # Numbered claim 1: "1. A compound of..." / "1. A pharmaceutical composition..."
    re.compile(
        r'^\s*1\.\s+A\s+(?:compound|composition|pharmaceutical|method|process|use|formulation)',
        re.MULTILINE,
    ),
    # "What is claimed is:" with flexible casing (OCR may produce mixed case)
    # Not anchored to start-of-line — two-column text may have garbage before it
    re.compile(r'What\s+is\s+claimed\s+is\s*:', re.MULTILINE | re.IGNORECASE),
]


def _is_examples_section_start(text: str, match: re.Match) -> bool:
    """Validate that an EXAMPLES match is the actual section start, not a reference.

    Patent Description sections often contain forward references like:
        "The following reaction schemes and Examples illustrate..."
        EXAMPLES
        "[0096] The following examples are meant to be illustrative..."

    The real Examples section start is distinguished by:
    - Specific compound/synthesis patterns nearby ("Example 1", compound names)
    - NOT just a passing reference to "examples" in description text

    Args:
        text: Full OCR text of the page.
        match: The regex match for the EXAMPLES heading.

    Returns:
        True if this appears to be the real Examples section start.
    """
    matched_text = match.group().strip()

    # Specific patterns like "Synthesis of Example 1" or "Example 1:" are
    # inherently reliable — these only appear in the actual Examples section
    if re.search(r'(?:Synthesis|Preparation)\s+of\s+Example', matched_text, re.IGNORECASE):
        return True
    if re.search(r'Example\s+1\s*[:.]\s*', matched_text):
        return True
    if re.search(r'EXAMPLE\s+1\b', matched_text):
        return True
    if re.search(r'EXEMPLARY\s+COMPOUNDS?', matched_text, re.IGNORECASE):
        return True

    # For generic "EXAMPLES" headers, check context to filter out
    # description-section forward references
    pos = match.start()
    context_after = text[pos:pos + 500]

    # Positive signal: compound synthesis language nearby
    if re.search(
        r'(?:synthesis|preparation|salt|acid|mmol|mg\b|mL\b|Step\s+1)',
        context_after, re.IGNORECASE
    ):
        return True

    # Negative signal: "illustrate", "are meant to be" suggest a
    # description-section forward reference, not the actual section
    if re.search(r'(?:illustrat|are\s+meant\s+to\s+be|following\s+examples?\s+are)',
                 context_after, re.IGNORECASE):
        return False

    return True


def _is_heading_context(text: str, match: re.Match, pattern: re.Pattern) -> bool:
    """Check whether a regex match appears in heading context (not inline).

    All-caps standalone patterns (EXAMPLES, CLAIMS, etc.) are trusted as-is
    since the regex already requires them on their own line. The heading-context
    check is only applied to mixed-case patterns like "Example 1:" which could
    appear as inline references in Description text.
    """
    # All-caps standalone headers are inherently safe — regex already enforces
    # start-of-line via ^ anchor. Two-column PDFs interleave text from adjacent
    # columns, so we check the first word of the match (not the full match,
    # which may contain merged lowercase column text).
    matched_text = match.group().strip()
    first_word = matched_text.split()[0] if matched_text else ""
    if first_word.isupper() and first_word.isalpha() and len(first_word) >= 4:
        return True
    if matched_text.upper().startswith("WHAT IS CLAIMED"):
        return True

    # For mixed-case patterns (e.g., "Example 1:"), check context
    pos = match.start()
    text_len = len(text)

    # Near top of page text
    if pos < text_len * 0.3:
        return True

    # Preceded by blank line or short line (section break)
    preceding = text[:pos]
    lines_before = preceding.rsplit('\n', 2)
    if len(lines_before) >= 2:
        prev_line = lines_before[-2].strip() if len(lines_before) > 1 else ""
        if len(prev_line) < 5:
            return True

    return False


class PatentSectionDetector:
    """Detect Examples and Claims section boundaries in patent PDFs."""

    def detect(
        self,
        pdf_path: str,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> SectionBounds:
        """Detect patent section boundaries using text extraction, then OCR fallback.

        Args:
            pdf_path: Path to the PDF file.
            progress_callback: Optional callback receiving status message strings.

        Returns:
            SectionBounds with detected page ranges.
        """
        # Tier 1: Text extraction
        if progress_callback:
            progress_callback("Detecting patent sections (text)...")
        bounds = self._detect_via_text(pdf_path)

        if bounds.is_valid and bounds.claims_start is not None:
            # Found both boundaries via text — done
            logger.info("Section detection (text): %s", bounds.summary())
            return bounds

        # Tier 2: OCR on sampled pages
        # Needed when: (a) no text found at all, or (b) text found Examples
        # but Claims are on image-based pages (mixed PDF)
        if HAS_TESSERACT:
            if progress_callback:
                progress_callback("Detecting patent sections (OCR)...")
            # Pass text bounds so OCR can skip Examples search if already found
            ocr_bounds = self._detect_via_ocr(
                pdf_path, bounds.total_pages, text_bounds=bounds
            )

            if bounds.is_valid and not ocr_bounds.is_valid:
                # Text found Examples, OCR found nothing new — use text result
                logger.info("Section detection (text, partial): %s", bounds.summary())
                return bounds

            if ocr_bounds.is_valid:
                # Merge: prefer text-based Examples if available (more precise)
                if bounds.is_valid:
                    ocr_bounds.examples_start = bounds.examples_start
                    ocr_bounds.detection_method = "text+ocr"
                logger.info("Section detection (%s): %s",
                            ocr_bounds.detection_method, ocr_bounds.summary())
                return ocr_bounds

        # Text found Examples but not Claims, no OCR available
        if bounds.is_valid:
            logger.info("Section detection (text, partial): %s", bounds.summary())
            return bounds

        # Tier 3: Claude Vision — most robust, works on any PDF
        if _has_claude_vision():
            if progress_callback:
                progress_callback("Detecting patent sections (Claude Vision)...")
            vision_bounds = self._detect_via_vision(pdf_path, bounds.total_pages)
            if vision_bounds.is_valid:
                logger.info("Section detection (vision): %s", vision_bounds.summary())
                return vision_bounds

        # Fallback: no sections detected
        logger.info("No patent sections detected in %s", pdf_path)
        return bounds

    def _detect_via_text(self, pdf_path: str) -> SectionBounds:
        """Tier 1: Use pdfplumber to extract text and search for section headers."""
        import pdfplumber

        bounds = SectionBounds()

        try:
            with pdfplumber.open(pdf_path) as pdf:
                bounds.total_pages = len(pdf.pages)

                for page_idx, page in enumerate(pdf.pages):
                    page_num = page_idx + 1  # 1-indexed

                    text = page.extract_text() or ""
                    if len(text.strip()) < 10:
                        continue

                    # Search for Examples start (take first match)
                    if bounds.examples_start is None:
                        for pattern in _EXAMPLES_PATTERNS:
                            m = pattern.search(text)
                            if m and _is_heading_context(text, m, pattern) \
                                    and _is_examples_section_start(text, m):
                                bounds.examples_start = page_num
                                break

                    # Search for Claims start (only after Examples)
                    if bounds.examples_start is not None and bounds.claims_start is None:
                        for pattern in _CLAIMS_PATTERNS:
                            m = pattern.search(text)
                            if m and _is_heading_context(text, m, pattern):
                                bounds.claims_start = page_num
                                break

                    # If we found both, stop early
                    if bounds.examples_start is not None and bounds.claims_start is not None:
                        break

        except Exception as e:
            logger.debug("Text-based section detection failed: %s", e)
            return bounds

        if bounds.is_valid:
            bounds.detection_method = "text"
        return bounds

    def _detect_via_ocr(
        self,
        pdf_path: str,
        total_pages: int,
        text_bounds: Optional[SectionBounds] = None,
    ) -> SectionBounds:
        """Tier 2: OCR sampled pages to find section headers in image-based PDFs.

        Two-phase approach for speed:
        1. Find Examples: scan forward every 3rd page (skipped if text_bounds
           already found Examples)
        2. Find Claims: scan backward from end every page in the last 20%
        Refines each boundary by checking adjacent pages.
        """
        import pypdfium2 as pdfium

        bounds = SectionBounds(total_pages=total_pages)

        # If text extraction already found Examples, carry it forward
        if text_bounds and text_bounds.examples_start is not None:
            bounds.examples_start = text_bounds.examples_start

        try:
            doc = pdfium.PdfDocument(pdf_path)
            if total_pages == 0:
                total_pages = len(doc)
                bounds.total_pages = total_pages

            # Phase 1: Find Examples start
            # Scan every page sequentially until we find the heading. At 100
            # DPI, each page takes ~0.5s to OCR. Examples is typically in the
            # first 20-50% of a patent. Skip the first 5% (cover/references).
            # (Skip entirely if already found via text extraction)
            if bounds.examples_start is None:
                skip_pages = max(3, int(total_pages * 0.05))
                for page_idx in range(skip_pages, total_pages):
                    text = self._ocr_page(doc, page_idx)
                    if not text:
                        continue

                    for pattern in _EXAMPLES_PATTERNS:
                        m = pattern.search(text)
                        if m and _is_examples_section_start(text, m):
                            bounds.examples_start = page_idx + 1  # 1-indexed
                            break

                    if bounds.examples_start is not None:
                        break

            # Phase 2: Find Claims start by scanning backward from end
            # Claims section is typically in the last ~10% of the patent.
            # Scan every page in the last 15% (small region), then every 3rd
            # page further back. This ensures we don't miss the Claims page.
            if bounds.examples_start is not None:
                claims_start_idx = max(
                    int(total_pages * 0.75),
                    bounds.examples_start  # Don't look before Examples
                )
                # Last 20% of pages: scan every page (guaranteed to find Claims)
                dense_start = max(claims_start_idx, int(total_pages * 0.80))
                # Sparse scan for earlier region
                scan_indices = list(range(total_pages - 1, dense_start - 1, -1))
                scan_indices += list(range(dense_start - 1, claims_start_idx - 1, -3))
                for page_idx in scan_indices:
                    text = self._ocr_page(doc, page_idx)
                    if not text:
                        continue

                    for pattern in _CLAIMS_PATTERNS:
                        m = pattern.search(text)
                        if m:
                            exact = self._refine_boundary(
                                doc, page_idx, _CLAIMS_PATTERNS, direction=-1
                            )
                            bounds.claims_start = exact
                            break

                    if bounds.claims_start is not None:
                        break

            doc.close()

        except Exception as e:
            logger.debug("OCR-based section detection failed: %s", e)
            return bounds

        if bounds.is_valid:
            bounds.detection_method = "ocr"
        return bounds

    def _ocr_page(self, doc, page_idx: int) -> str:
        """Render and OCR a page for section header detection.

        Uses 150 DPI — sufficient for recognizing section headers in both
        text-based and image-based/scanned PDFs. Lower DPI (72-100) produces
        garbled text on scanned patents.

        Args:
            doc: pypdfium2 PdfDocument.
            page_idx: 0-indexed page index.

        Returns:
            Extracted text, or empty string on failure.
        """
        try:
            page = doc[page_idx]

            # 150 DPI — reliable OCR on scanned/image-based patents
            scale = 150 / 72
            bitmap = page.render(scale=scale, rotation=0)
            pil_image = bitmap.to_pil()

            text = pytesseract.image_to_string(pil_image)
            return text

        except Exception as e:
            logger.debug("OCR failed for page %d: %s", page_idx + 1, e)
            return ""

    def _refine_boundary(
        self,
        doc,
        approx_page_idx: int,
        patterns: list,
        direction: int = -1,
    ) -> int:
        """Check adjacent pages to find the exact section boundary.

        After OCR sampling finds a match on e.g. page 60, check pages 56-59
        to see if the section actually starts earlier.

        Args:
            doc: pypdfium2 PdfDocument.
            approx_page_idx: 0-indexed page where the match was found.
            patterns: List of regex patterns to search for.
            direction: -1 to search backwards (find earliest match).

        Returns:
            1-indexed page number of the refined boundary.
        """
        best_page = approx_page_idx + 1  # 1-indexed, start with the known match

        if direction == -1:
            # Check up to 4 preceding pages
            start = max(0, approx_page_idx - 4)
            check_range = range(start, approx_page_idx)
        else:
            # Check up to 4 following pages
            end = min(len(doc), approx_page_idx + 5)
            check_range = range(approx_page_idx + 1, end)

        for idx in check_range:
            text = self._ocr_page(doc, idx)
            if not text:
                continue
            for pattern in patterns:
                if pattern.search(text):
                    candidate = idx + 1  # 1-indexed
                    if direction == -1:
                        best_page = min(best_page, candidate)
                    else:
                        best_page = max(best_page, candidate)
                    break

        return best_page

    # ── Tier 3: Claude Vision ────────────────────────────────

    _VISION_MODEL = "claude-haiku-4-5-20251001"

    _VISION_PROMPT = """\
Analyze this patent PDF page. Which section of the patent does this page belong to?

Respond with JSON only:
{
  "section": "description" | "examples" | "claims" | "tables" | "other",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}

Section definitions:
- "description": Detailed description of the invention, general formulas, Markush structures with R-groups, generic reaction schemes, background, definitions. Pages with Formula (I), Formula (II), or structures with variable groups (R1, R2, X, Y).
- "examples": Specific synthesis examples (Example 1, Example 2...), preparation of named compounds with specific substituents, step-by-step reaction procedures with actual reagents and quantities.
- "claims": Patent claims section ("What is claimed is:", numbered claims like "1. A compound of...").
- "tables": SAR tables, biological data tables (IC50, EC50), compound property tables with columns of numeric data.
- "other": Title page, abstract, references, sequence listings, drawings legend."""

    def _detect_via_vision(
        self,
        pdf_path: str,
        total_pages: int,
    ) -> SectionBounds:
        """Tier 3: Use Claude Vision to classify sampled pages and find section boundaries.

        Samples ~8-12 pages spread across the PDF, asks Claude what section each
        belongs to, then infers boundary positions. Cost: ~$0.05-0.10 per patent.

        Args:
            pdf_path: Path to the PDF file.
            total_pages: Total number of pages.

        Returns:
            SectionBounds with detected page ranges.
        """
        import anthropic
        import pypdfium2 as pdfium

        bounds = SectionBounds(total_pages=total_pages)

        try:
            client = anthropic.Anthropic()
            doc = pdfium.PdfDocument(pdf_path)
            if total_pages == 0:
                total_pages = len(doc)
                bounds.total_pages = total_pages

            # Sample pages: spread across the PDF to find transitions
            # More density around the typical Examples start (20-50% mark)
            # and Claims start (80-95% mark)
            sample_indices = set()
            # Early pages (description)
            sample_indices.add(max(0, int(total_pages * 0.05)))
            sample_indices.add(int(total_pages * 0.15))
            # Mid pages (likely near Examples start)
            for frac in [0.25, 0.30, 0.35, 0.40, 0.50]:
                sample_indices.add(int(total_pages * frac))
            # Late pages (near Claims)
            for frac in [0.70, 0.80, 0.90, 0.95]:
                sample_indices.add(int(total_pages * frac))
            # Clamp to valid range
            sample_indices = sorted(
                idx for idx in sample_indices if 0 <= idx < total_pages
            )

            # Classify each sampled page
            page_sections = {}  # page_idx -> section string
            for page_idx in sample_indices:
                section = self._vision_classify_page(client, doc, page_idx)
                if section:
                    page_sections[page_idx] = section
                    logger.debug(
                        "Vision: page %d -> %s", page_idx + 1, section
                    )

            doc.close()

            if not page_sections:
                return bounds

            # Find the transition points
            # Examples start: first page classified as "examples" or "tables"
            # (tables after examples are usually SAR data tables)
            examples_candidates = [
                idx for idx, s in page_sections.items()
                if s in ("examples", "tables")
            ]
            claims_candidates = [
                idx for idx, s in page_sections.items()
                if s == "claims"
            ]

            if examples_candidates:
                approx_start = min(examples_candidates)
                # Binary search to refine: check pages between the last
                # "description" page and the first "examples" page
                desc_pages = [
                    idx for idx, s in page_sections.items()
                    if s == "description" and idx < approx_start
                ]
                search_start = max(desc_pages) + 1 if desc_pages else max(0, approx_start - 5)
                # Refine by checking a few pages in the gap
                best_start = approx_start
                for page_idx in range(search_start, approx_start):
                    section = self._vision_classify_page(client, pdfium.PdfDocument(pdf_path), page_idx)
                    if section in ("examples", "tables"):
                        best_start = min(best_start, page_idx)
                        break
                bounds.examples_start = best_start + 1  # 1-indexed

            if claims_candidates:
                approx_claims = min(claims_candidates)
                bounds.claims_start = approx_claims + 1  # 1-indexed

            if bounds.is_valid:
                bounds.detection_method = "vision"

        except Exception as e:
            logger.debug("Vision-based section detection failed: %s", e)

        return bounds

    def _vision_classify_page(self, client, doc, page_idx: int) -> Optional[str]:
        """Classify a single page using Claude Vision.

        Args:
            client: Anthropic client instance.
            doc: pypdfium2 PdfDocument.
            page_idx: 0-indexed page index.

        Returns:
            Section string ("description", "examples", "claims", "tables", "other")
            or None on failure.
        """
        try:
            page = doc[page_idx]
            # 100 DPI is enough for Claude to read page content
            bitmap = page.render(scale=100 / 72, rotation=0)
            pil_image = bitmap.to_pil()

            # Encode image
            buf = io.BytesIO()
            pil_image.save(buf, format="PNG")
            image_data = base64.standard_b64encode(buf.getvalue()).decode("ascii")

            response = client.messages.create(
                model=self._VISION_MODEL,
                max_tokens=200,
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
                                "text": self._VISION_PROMPT,
                            },
                        ],
                    }
                ],
            )

            text = response.content[0].text.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1])

            data = json.loads(text)
            section = data.get("section", "other")
            if section in ("description", "examples", "claims", "tables", "other"):
                return section
            return "other"

        except Exception as e:
            logger.debug("Vision classify page %d failed: %s", page_idx + 1, e)
            return None
