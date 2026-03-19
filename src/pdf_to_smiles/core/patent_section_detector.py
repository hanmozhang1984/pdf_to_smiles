"""Detect patent section boundaries (Examples, Claims) to limit processing scope.

Chemistry patents follow a predictable structure:
Abstract -> Background -> Detailed Description -> Examples -> Bio Data -> Claims

By detecting where Examples starts and Claims begins, we can skip 30-50% of pages
that contain Markush structures (Description) or claim text (Claims).

Two-tier detection:
  Tier 1: pdfplumber text extraction (fast, free, works on text-based PDFs)
  Tier 2: pytesseract OCR on sampled pages (slower, works on image-based PDFs)
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Try to import pytesseract for OCR fallback
try:
    import pytesseract
    from PIL import Image
    from ..utils.paths import configure_tesseract
    HAS_TESSERACT = configure_tesseract()
except ImportError:
    HAS_TESSERACT = False


@dataclass
class SectionBounds:
    """Page boundaries for patent sections."""

    examples_start: Optional[int] = None  # 1-indexed page number
    claims_start: Optional[int] = None    # 1-indexed page number
    total_pages: int = 0
    detection_method: str = "none"        # "text", "ocr", or "none"

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
    - NO bracketed paragraph numbers ([0096]) in nearby lines
    - Often followed by "Example 1" or specific compound synthesis text

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
    # Get ~500 chars after the match for context
    context_after = text[pos:pos + 500]

    # Negative signal: bracketed paragraph numbers indicate Description section
    # e.g., [0095], [0096] — these are patent paragraph numbering
    if re.search(r'\[\d{4,}\]', context_after):
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
        """Render and OCR a page at low DPI for section header detection.

        Uses 72 DPI (1:1 with PDF points) — sufficient for recognizing large
        section header text (EXAMPLES, CLAIMS, etc.) while being fast.

        Args:
            doc: pypdfium2 PdfDocument.
            page_idx: 0-indexed page index.

        Returns:
            Extracted text, or empty string on failure.
        """
        try:
            page = doc[page_idx]

            # 100 DPI — fast rendering while maintaining OCR accuracy
            # for scanned/image-based pages (72 DPI too low for these)
            scale = 100 / 72
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
