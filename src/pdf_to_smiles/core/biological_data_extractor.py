"""Biological data extraction from patent PDF tables.

Extracts assay data (IC50, EC50, Ki, Kd) from tables and associates
them with compound numbers.

Uses pdfplumber (MIT) for table/text extraction and pypdfium2 (Apache 2.0) for OCR.
"""

import re
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import pdfplumber
import pypdfium2 as pdfium

# Try to import OCR libraries and configure Tesseract path
try:
    import pytesseract
    from PIL import Image
    from ..utils.paths import configure_tesseract
    HAS_OCR = configure_tesseract()
except ImportError:
    HAS_OCR = False


@dataclass
class BiologicalData:
    """Biological assay data for a compound."""
    compound_id: str
    ic50: Optional[str] = None
    ec50: Optional[str] = None
    ki: Optional[str] = None
    kd: Optional[str] = None
    # Store additional assay data as dict
    other_assays: Dict[str, str] = field(default_factory=dict)
    # Track if this was matched to a structure
    matched: bool = False


class BiologicalDataExtractor:
    """Extracts biological assay data from patent PDF tables.

    Searches for tables containing IC50, EC50, Ki, Kd values and
    associates them with compound numbers.
    """

    # Keywords that indicate biological data tables
    TABLE_KEYWORDS = [
        'ic50', 'ec50', 'ki', 'kd', 'gi50', 'cc50', 'ed50', 'ld50',
        'emax', 'ec90', 'ic90', 'mic', 'mec',
        'inhibition', 'activity', 'potency',
        'binding', 'affinity', 'assay',
        'nm', 'μm', 'um', 'pm', 'mm',  # Units often indicate assay data
        'cells', 'cell line'  # Cell-based assays
    ]

    # Keywords that indicate NMR/analytical/synthesis characterization (NOT bio data)
    NMR_ANALYTICAL_KEYWORDS = [
        'nmr', 'mhz', 'dmso', 'ppm', 'esi', 'lcms', 'hplc', 'sfc',
        'm/z', 'cdcl', 'methanol-d', 'optical rotation', 'chiral',
        'yield', '1h nmr', '13c nmr', '19f nmr', 'iupac',
        'analytical data', 'physicochemical',
        # Synthesis procedure keywords
        'chromatog', 'etoac', 'purificatio', 'heated at',
        'flash column', 'silica gel',
    ]

    # Patterns for assay column headers - more flexible
    ASSAY_PATTERNS = {
        'ic50': re.compile(r'ic\s*[-_]?\s*50', re.IGNORECASE),
        'ec50': re.compile(r'ec\s*[-_]?\s*50', re.IGNORECASE),
        'ki': re.compile(r'\bki\b', re.IGNORECASE),
        'kd': re.compile(r'\bkd\b', re.IGNORECASE),
    }

    # Pattern for compound number column
    COMPOUND_COL_PATTERNS = [
        re.compile(r'(?:compound|cpd|cmpd|no\.?|#|example|ex\.?|structure)', re.IGNORECASE),
    ]

    # Pattern for compound ID values - more flexible
    COMPOUND_ID_PATTERN = re.compile(
        r'[\s\(\[]*(\d{1,4}[a-zA-Z]?)[\s\)\]]*'
    )

    # Pattern for numeric values with optional units and modifiers
    VALUE_PATTERN = re.compile(
        r'([<>≤≥±~]?\s*\d+\.?\d*)\s*(nM|μM|uM|mM|pM|%)?',
        re.IGNORECASE
    )

    # Pattern for symbolic values - more flexible
    SYMBOL_PATTERN = re.compile(
        r'([+＋]{1,5}|[-−–—]{1,5}|ND|NA|NT|n\.?d\.?|n\.?a\.?|inactive|active)',
        re.IGNORECASE
    )

    def __init__(self):
        """Initialize the biological data extractor."""
        self._debug_info = []
        self._last_extracted_data: Dict[str, BiologicalData] = {}
        self._unmatched_compounds: Dict[str, BiologicalData] = {}
        self._cached_header_columns: List[str] = []  # Remember header across pages

    @staticmethod
    def _fix_ocr_compound_id(text: str) -> str:
        """Fix common OCR digit↔letter confusions in compound IDs.

        OCR often misreads: B↔3, l/I↔1, s/S↔5, O↔0, Z↔2, g↔9, b↔6.
        Only fix if the result would be all-numeric (with optional dash).
        """
        # Map of common OCR letter→digit confusions
        ocr_fixes = {
            'B': '3', 'b': '6',
            'l': '1', 'I': '1', '|': '1',
            's': '5', 'S': '5',
            'O': '0', 'o': '0',
            'Z': '2', 'z': '2',
            'g': '9', 'q': '9',
        }

        # Only attempt fix if the text has mixed digits and letters
        has_digit = any(c.isdigit() for c in text.replace('-', ''))
        has_letter = any(c.isalpha() for c in text)
        if not (has_digit and has_letter):
            return text

        # Try replacing each letter with its digit equivalent
        fixed = []
        for c in text:
            if c in ocr_fixes:
                fixed.append(ocr_fixes[c])
            else:
                fixed.append(c)
        result = ''.join(fixed)

        # Only accept if result is now all-numeric (with optional dashes)
        cleaned = result.replace('-', '')
        if cleaned.isdigit():
            return result

        return text

    def _filter_outlier_compound_ids(
        self, examples: List[Tuple[str, int]]
    ) -> List[Tuple[str, int]]:
        """Filter out data values misidentified as compound IDs.

        When OCR processes a two-column table, data values from the right column
        can appear in the left margin area and get treated as example numbers.
        E.g., "680", "480", "190" are assay values, not compound IDs.

        Strategy: Use IQR (interquartile range) to detect outlier IDs.
        IDs above Q3 + 1.5*IQR are likely data values, not compound IDs.
        Protect consecutive ID clusters (3+ sequential numbers) from filtering.
        """
        if len(examples) < 5:
            return examples

        # Get numeric values of IDs (ignoring dash-separated)
        numeric_ids = []
        for num_str, _ in examples:
            try:
                base = num_str.split('-')[0]
                numeric_ids.append(int(base))
            except ValueError:
                pass

        if len(numeric_ids) < 5:
            return examples

        numeric_ids_sorted = sorted(numeric_ids)
        n = len(numeric_ids_sorted)
        q1 = numeric_ids_sorted[n // 4]
        q3 = numeric_ids_sorted[(3 * n) // 4]
        iqr = q3 - q1

        if iqr <= 0:
            # All IDs are very close together, use simple max + buffer
            threshold = q3 + 50
        else:
            threshold = q3 + 1.5 * iqr

        # Find IDs above threshold that form consecutive clusters (likely valid)
        # Data values are typically isolated; real compound IDs come in sequences
        above_threshold = sorted(set(x for x in numeric_ids if x > threshold))
        protected_ids = set()
        if above_threshold:
            # Find clusters of consecutive IDs (allowing gap of 2 for missing IDs)
            clusters = []
            current_cluster = [above_threshold[0]]
            for i in range(1, len(above_threshold)):
                if above_threshold[i] - above_threshold[i - 1] <= 2:
                    current_cluster.append(above_threshold[i])
                else:
                    clusters.append(current_cluster)
                    current_cluster = [above_threshold[i]]
            clusters.append(current_cluster)

            # Protect clusters of 3+ consecutive IDs
            for cluster in clusters:
                if len(cluster) >= 3:
                    protected_ids.update(cluster)

        filtered = []
        for num_str, y in examples:
            try:
                base = int(num_str.split('-')[0])
                if base <= threshold or base in protected_ids:
                    filtered.append((num_str, y))
                else:
                    self._debug_info.append(f"  Filtered outlier ID: {num_str} (threshold={threshold:.0f})")
            except ValueError:
                filtered.append((num_str, y))

        return filtered

    def _text_looks_two_column(self, text: str) -> bool:
        """Check if text appears to be from a two-column table layout.

        Two-column bio data tables have patterns like:
        - "Example...Example" on the same line
        - "TABLE X...TABLE X" (two TABLE references)
        - Repeated assay columns (Ki...Ki, IC50...IC50)
        """
        patterns = [
            r'Example.*Example',
            r'TABLE\s+\d+.*TABLE\s+\d+',
            r'IC\s*50.*IC\s*50',
            r'EC\s*50.*EC\s*50',
            r'K\s*[;,i].*K\s*[;,i].*K\s*[;,i].*K\s*[;,i]',  # Multiple Ki columns
            r'Assay\s+\d.*Assay\s+\d.*Assay\s+\d.*Assay\s+\d',
            r'Exam-.*Exam-',
        ]
        for line in text.split('\n'):
            for p in patterns:
                if re.search(p, line, re.IGNORECASE):
                    return True
        return False

    def _is_nmr_analytical_page(self, text: str) -> bool:
        """Check if page text is primarily NMR/analytical characterization data.

        These pages contain structure tables with NMR spectra, LCMS data, etc.
        but no biological assay data. Extracting from them produces false positives
        like ic50=400 M (from "400 MHz").
        """
        text_lower = text.lower()
        nmr_count = sum(1 for kw in self.NMR_ANALYTICAL_KEYWORDS if kw in text_lower)
        # Also check for NMR-specific patterns
        if re.search(r'\d+\s*MHz', text):
            nmr_count += 2
        if re.search(r'[JjδΔ]\s*=\s*[\d.]+\s*Hz', text):
            nmr_count += 2
        if re.search(r'\bm/z\b', text):
            nmr_count += 1
        # Check for actual bio keywords
        bio_count = 0
        bio_keywords = ['ic50', 'ec50', 'ki ', 'kd ', 'gi50', 'emax',
                       'inhibition', 'activity', 'biological', 'potency',
                       'assay 1', 'assay 2', 'antagonist']
        for kw in bio_keywords:
            if kw in text_lower:
                bio_count += 1
        # If NMR keywords dominate, it's an analytical page
        return nmr_count >= 3 and bio_count == 0

    def _correct_ocr_assay_name(self, name: str) -> str:
        """Correct common OCR errors in assay names.

        Common OCR misreads:
        - 5 → s, S
        - 0 → o, O
        - 1 → l, I
        - So IC50 becomes ICso, ICs0, lC50, etc.

        Args:
            name: Raw OCR'd assay name.

        Returns:
            Corrected assay name.
        """
        if not name:
            return name

        corrected = name

        # Patterns to correct: GIso, ICso, ECso, Glso, lCso, etc. → GI50, IC50, EC50
        # Also handle: GIs0, ICs0, ECs0 (zero as letter o)
        ocr_corrections = [
            # IC50 variants
            (r'\b[Il1][Cc]\s*[Ss5][Oo0]\b', 'IC50'),
            (r'\b[Il1][Cc]\s*[-_]?\s*[Ss5][Oo0]\b', 'IC50'),
            (r'\b[Il1][Cc][Ss5]\b', 'IC50'),  # ICs, IC5 (truncated)
            # EC50 variants
            (r'\b[Ee][Cc]\s*[Ss5][Oo0]\b', 'EC50'),
            (r'\b[Ee][Cc]\s*[-_]?\s*[Ss5][Oo0]\b', 'EC50'),
            # GI50 variants (common: Glso, GIso, Gls0)
            (r'\b[Gg][Il1]\s*[Ss5][Oo0]\b', 'GI50'),
            (r'\b[Gg][Il1]\s*[-_]?\s*[Ss5][Oo0]\b', 'GI50'),
            # CC50 variants
            (r'\b[Cc][Cc]\s*[Ss5][Oo0]\b', 'CC50'),
            # ED50 variants
            (r'\b[Ee][Dd]\s*[Ss5][Oo0]\b', 'ED50'),
            # LD50 variants
            (r'\b[Ll][Dd]\s*[Ss5][Oo0]\b', 'LD50'),
            # IC90, EC90 variants
            (r'\b[Il1][Cc]\s*[9gq][Oo0]\b', 'IC90'),
            (r'\b[Ee][Cc]\s*[9gq][Oo0]\b', 'EC90'),
            # Emax variants (Ernax, Emox)
            (r'\b[Ee]\s*[Mm]\s*[Aa]\s*[Xx]\b', 'Emax'),
            (r'\b[Ee][Mm][Oo0][Xx]\b', 'Emax'),
            (r'\b[Ee][Rr][Nn][Aa][Xx]\b', 'Emax'),
        ]

        for pattern, replacement in ocr_corrections:
            corrected = re.sub(pattern, replacement, corrected)

        # If correction was made, log it
        if corrected != name:
            self._debug_info.append(f"  OCR correction: '{name}' -> '{corrected}'")

        return corrected

    def _clean_header_columns(self, columns: List[str]) -> List[str]:
        """Clean up header column names after merging/extraction.

        Fixes:
        - Deduplicate repeated column names (e.g., "AcCoA (nM) AcCoA (nM)" → "AcCoA (nM)")
        - Fix truncated "Exam-" headers
        - Strip compound column ("No.", "Example") from header columns
        - Clean garbled OCR text
        """
        cleaned = []
        for col in columns:
            c = col.strip()
            if not c:
                continue

            # Deduplicate repeated substrings within a column name
            # e.g., "AcCoA (nM) AcCoA (nM) AcCoA (nM)" → "AcCoA (nM)"
            words_list = c.split()
            if len(words_list) >= 4:
                # Try to find the shortest repeating unit
                for unit_len in range(1, len(words_list) // 2 + 1):
                    unit = ' '.join(words_list[:unit_len])
                    # Check if the full string is just this unit repeated
                    full_repeated = ' '.join([unit] * (len(words_list) // unit_len))
                    remaining = ' '.join(words_list[unit_len * (len(words_list) // unit_len):])
                    if full_repeated == c or (remaining and c.startswith(full_repeated)):
                        c = unit
                        break

            # Fix "Exam- N" → "Assay N" (truncated "Example" header used as column label)
            exam_match = re.match(r'^Exam-?\s*(\d+)$', c, re.IGNORECASE)
            if exam_match:
                c = f"Assay {exam_match.group(1)}"

            # Fix garbled compound+header merged text
            # "mple IC50 (nM)! count Compound Name" → "IC50 (nM)"
            if re.search(r'mple|xample|compound\s+name', c, re.IGNORECASE):
                # Extract just the assay part
                assay_match = re.search(r'((?:IC|EC|GI|CC)\s*50\s*(?:\([^)]*\))?)', c, re.IGNORECASE)
                if assay_match:
                    c = assay_match.group(1).strip()
                else:
                    # Just strip the "Example"/"Compound" part
                    c = re.sub(r'(?:exa)?mple|compound\s*name|[!|]', '', c, flags=re.IGNORECASE).strip()
                    c = re.sub(r'^\s*count\s*', '', c, flags=re.IGNORECASE).strip()

            # Fix "No. (units) (units)" — compound ID column merged with unit rows
            # Only strip when "No." is followed by multiple unit groups like "(mM) (nM)"
            no_units_match = re.match(r'^No\.?\s+(\([^)]+\)\s+\([^)]+\))', c, re.IGNORECASE)
            if no_units_match:
                # Multiple units after "No." indicates merged compound+data headers; skip
                continue
            # "No." alone as column header — keep it (may be compound ID column in header)
            # Don't strip it as it maintains column alignment

            # Pure column number (just "1", "2", etc.) — rename to Assay_N
            if re.match(r'^\d{1,2}$', c):
                c = f"Assay_{c}"

            # Truncate overly long column names (likely full sentences from OCR)
            # Try to extract a meaningful assay name from the text
            if len(c) > 60:
                # Try to find an assay keyword in the long text
                assay_match = re.search(
                    r'((?:IC|EC|GI|CC)\s*50\s*(?:\([^)]*\))?)',
                    c, re.IGNORECASE
                )
                if assay_match:
                    c = assay_match.group(1).strip()
                else:
                    # Look for common assay-related terms to build a shorter name
                    for keyword in ['inhibition', 'activity', 'binding', 'potency',
                                    'competition', 'agonist', 'antagonist']:
                        if keyword in c.lower():
                            # Use "% inhibition" or similar
                            c = f"% {keyword}"
                            break
                    else:
                        # Last resort: truncate to first ~40 chars
                        c = c[:40].rsplit(' ', 1)[0]

            if c:
                cleaned.append(c)

        return cleaned

    def extract_from_pdf_path(self, pdf_path: str, page_filter=None) -> Dict[str, BiologicalData]:
        """Extract biological data from all tables in a PDF.

        Args:
            pdf_path: Path to the PDF file.
            page_filter: Optional set of 1-indexed page numbers to process.
                         None or empty means all pages.

        Returns:
            Dictionary mapping compound_id to BiologicalData.
        """
        all_data = {}
        self._debug_info = []
        self._unmatched_compounds = {}
        self._cached_header_columns = []  # Reset header cache for new PDF
        self._next_positional_id = 1  # Reset positional ID counter for new PDF

        try:
            # Open with pdfplumber for tables and text
            with pdfplumber.open(pdf_path) as plumber_doc:
                # Also open with pypdfium2 for OCR rendering
                pdfium_doc = pdfium.PdfDocument(pdf_path)

                for page_num in range(len(plumber_doc.pages)):
                    # Skip pages not in filter (page_num is 0-indexed here)
                    if page_filter and (page_num + 1) not in page_filter:
                        continue

                    plumber_page = plumber_doc.pages[page_num]
                    pdfium_page = pdfium_doc[page_num]
                    page_data = self._extract_from_page(
                        plumber_page, pdfium_page, page_num + 1
                    )

                    # Merge page data, preferring non-None values
                    for compound_id, bio_data in page_data.items():
                        if compound_id in all_data:
                            self._merge_bio_data(all_data[compound_id], bio_data)
                        else:
                            all_data[compound_id] = bio_data

                pdfium_doc.close()

        except Exception as e:
            self._debug_info.append(f"Error extracting from PDF: {e}")

        # Store for later reference
        self._last_extracted_data = all_data.copy()

        # Debug: Log final extraction summary
        self._debug_info.append(f"\n=== EXTRACTION SUMMARY ===")
        self._debug_info.append(f"Total compounds extracted: {len(all_data)}")
        if all_data:
            for cid, bd in sorted(all_data.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 999):
                assays_str = ', '.join(f"{k}={v}" for k, v in list(bd.other_assays.items())[:3])
                self._debug_info.append(f"  Compound {cid}: {assays_str}")

        return all_data

    # Legacy method for compatibility
    def extract_from_pdf(self, pdf_document, page_filter=None) -> Dict[str, BiologicalData]:
        """Extract biological data - legacy interface.

        Args:
            pdf_document: Can be a file path (str) or a legacy document object.
            page_filter: Optional set of 1-indexed page numbers to process.

        Returns:
            Dictionary mapping compound_id to BiologicalData.
        """
        # If it's a string path, use the new method
        if isinstance(pdf_document, str):
            return self.extract_from_pdf_path(pdf_document, page_filter=page_filter)

        # Try to get the file path from the document object
        # This handles both old PyMuPDF and new scenarios
        if hasattr(pdf_document, 'name'):
            return self.extract_from_pdf_path(pdf_document.name, page_filter=page_filter)

        # Fallback: return empty data
        self._debug_info.append("Cannot extract: unknown document type")
        return {}

    def get_unmatched_compounds(self) -> Dict[str, BiologicalData]:
        """Get compounds that were extracted but not matched to structures."""
        return {k: v for k, v in self._last_extracted_data.items() if not v.matched}

    def mark_as_matched(self, compound_id: str) -> None:
        """Mark a compound as matched to a structure."""
        if compound_id in self._last_extracted_data:
            self._last_extracted_data[compound_id].matched = True

    def get_all_extracted_compounds(self) -> List[str]:
        """Get list of all compound IDs that were extracted."""
        return list(self._last_extracted_data.keys())

    def _extract_from_page(
        self,
        plumber_page,
        pdfium_page,
        page_num: int
    ) -> Dict[str, BiologicalData]:
        """Extract biological data from tables on a single page."""
        data = {}

        try:
            # Get tables from the page using pdfplumber
            tables = plumber_page.extract_tables()
            self._debug_info.append(f"Page {page_num}: Found {len(tables)} tables")

            for table_idx, table_data in enumerate(tables):
                if not table_data or len(table_data) < 2:
                    continue

                self._debug_info.append(
                    f"  Table {table_idx + 1}: {len(table_data)} rows, "
                    f"{len(table_data[0]) if table_data else 0} cols"
                )
                self._debug_info.append(f"    Header: {table_data[0]}")

                if self._is_biological_table_data(table_data):
                    self._debug_info.append(f"    -> Identified as biological data table")
                    parsed = self._parse_biological_table_data(table_data)
                    self._debug_info.append(f"    -> Extracted {len(parsed)} compounds")

                    for compound_id, bio_data in parsed.items():
                        if compound_id in data:
                            self._merge_bio_data(data[compound_id], bio_data)
                        else:
                            data[compound_id] = bio_data

            # If no tables found or no data extracted, try text-based extraction
            if not data:
                self._debug_info.append(f"  No table data found, trying text-based extraction...")
                text_data = self._extract_from_page_text(plumber_page, page_num)
                data.update(text_data)

            # If still no data, try OCR-based extraction (for image-based tables)
            if not data and HAS_OCR:
                self._debug_info.append(f"  No text data found, trying OCR extraction...")
                ocr_data = self._extract_from_page_ocr(pdfium_page, page_num)
                data.update(ocr_data)
            elif not data and not HAS_OCR:
                self._debug_info.append(f"  OCR not available (pytesseract not installed)")

        except Exception as e:
            self._debug_info.append(f"Page {page_num}: Error - {e}")

        return data

    def _extract_from_page_text(self, plumber_page, page_num: int) -> Dict[str, BiologicalData]:
        """Extract biological data by parsing page text directly."""
        data = {}

        try:
            # Get all text from the page
            text = plumber_page.extract_text() or ""
            self._debug_info.append(f"  Page text length: {len(text)} chars")

            # Show a sample of the text for debugging
            sample = text[:500].replace('\n', ' | ')
            self._debug_info.append(f"  Text sample: {sample}...")

            # Skip NMR/analytical characterization pages
            if self._is_nmr_analytical_page(text):
                self._debug_info.append(f"  Skipping NMR/analytical page (no bio data)")
                return data

            # If this looks like a two-column table, skip text extraction
            # and let the OCR pipeline handle it (it has two-column splitting)
            if self._text_looks_two_column(text):
                self._debug_info.append(f"  Two-column layout detected in text, deferring to OCR")
                return data

            # Look for tabular patterns - compound number followed by symbols or values
            lines = text.split('\n')

            # First, find lines that look like headers (contain IC50, EC50, etc.)
            header_line_idx = None
            assay_columns = []
            for idx, line in enumerate(lines):
                line_lower = line.lower()
                if any(kw in line_lower for kw in ['ic50', 'ec50', 'ki', 'kd', 'inhibition', 'activity']):
                    header_line_idx = idx
                    self._debug_info.append(f"  Potential header at line {idx}: {line.strip()}")
                    for assay in ['ic50', 'ec50', 'ki', 'kd']:
                        if assay in line_lower:
                            assay_columns.append(assay)
                    break

            # Pattern: Look for lines that start with a number (compound ID) followed by symbols
            compound_pattern = re.compile(r'^\s*(\d{1,4}(?:-\d{1,4})?[a-zA-Z]?)\s+(.+)$')
            symbol_pattern = re.compile(r'[+＋]{1,5}|[-−–—]{1,5}|[\d.]+\s*[nμump]?M|ND|NA|NT', re.IGNORECASE)

            for line in lines:
                match = compound_pattern.match(line)
                if match:
                    compound_id = self._fix_ocr_compound_id(match.group(1))
                    rest = match.group(2)

                    # Filter out NMR notation (1H, 2H, 3H, 4H)
                    if re.match(r'^\d[HhDd]$', compound_id):
                        continue

                    values = symbol_pattern.findall(rest)
                    if values:
                        self._debug_info.append(f"  Found compound {compound_id}: {values}")

                        if compound_id not in data:
                            data[compound_id] = BiologicalData(compound_id=compound_id)

                        if assay_columns:
                            for i, assay in enumerate(assay_columns):
                                if i < len(values):
                                    setattr(data[compound_id], assay, values[i])
                        elif values:
                            data[compound_id].ic50 = values[0]

            self._debug_info.append(f"  Text extraction found {len(data)} compounds")

        except Exception as e:
            self._debug_info.append(f"  Text extraction error: {e}")

        return data

    def _normalize_plus_symbols(self, text: str) -> str:
        """Normalize OCR misreads of + symbols.

        Common OCR errors for ++++: +444, +44+, H+, t+, a+, H++, 4+, etc.

        Args:
            text: Raw OCR text.

        Returns:
            Text with + symbols cleaned up.
        """
        cleaned = text
        # If text is a mix of + and 4 characters (e.g., +444, +4+, 4++4),
        # treat all 4s as + since they look similar in many fonts
        if re.match(r'^[+＋4]+$', cleaned) and len(cleaned) >= 2:
            plus_count = len(cleaned)
            return '+' * min(plus_count, 5)
        # Replace common OCR misreads adjacent to + signs
        cleaned = re.sub(r'(?<=[+])4(?=[+])', '+', cleaned)  # +4+ -> +++
        cleaned = re.sub(r'(?<=[+])44(?=[+])', '++', cleaned)  # +44+ -> ++++
        cleaned = re.sub(r'4([+]{2,})', lambda m: '+' + m.group(1), cleaned)  # 4++ -> +++
        cleaned = re.sub(r'([+]{2,})4', lambda m: m.group(1) + '+', cleaned)  # ++4 -> +++
        cleaned = re.sub(r'[HhTtAa]([+]+)', lambda m: m.group(1), cleaned)  # H+ -> +, t++ -> ++
        cleaned = re.sub(r'([+]+)[HhTtAa]', lambda m: m.group(1), cleaned)  # +H -> +, ++t -> ++
        # Collapse runs of + with interspersed 4s: +4+4+ -> +++++
        cleaned = re.sub(r'[+4]{3,}', lambda m: '+' * sum(1 for c in m.group() if c in '+4'), cleaned)
        return cleaned

    def _extract_patent_table_columns(self, pil_image, page_num: int) -> Dict[str, BiologicalData]:
        """Extract bio data from patent tables using full-page OCR with position filtering.

        Targets the common patent table layout: Example | Structure (image) | IC50
        Runs OCR on the full page at native resolution, finds the IC50 column
        header position, then extracts example numbers from the left margin and
        activity values from the IC50 column, matching by Y-position.

        Args:
            pil_image: PIL Image of the page (already at 200dpi rendering).
            page_num: Page number for debugging.

        Returns:
            Dictionary mapping compound_id to BiologicalData.
        """
        data = {}

        try:
            width, height = pil_image.size

            # Full-page OCR at native resolution (200dpi is sufficient)
            ocr_data = pytesseract.image_to_data(
                pil_image, output_type=pytesseract.Output.DICT
            )

            # First pass: find all words with positions (low conf threshold
            # to catch misread + symbols like '+444' at conf=8)
            words = []
            n_boxes = len(ocr_data['text'])
            for i in range(n_boxes):
                text = str(ocr_data['text'][i]).strip()
                conf = int(ocr_data['conf'][i]) if ocr_data['conf'][i] != '-1' else 0
                if not text:
                    continue
                words.append({
                    'text': text,
                    'x': ocr_data['left'][i],
                    'y': ocr_data['top'][i],
                    'w': ocr_data['width'][i],
                    'h': ocr_data['height'][i],
                    'conf': conf,
                    'x_pct': ocr_data['left'][i] / width * 100,
                })

            if not words:
                return data

            # Find the IC50/assay column header and its X position
            assay_header_x = None
            assay_name = "IC50"

            # Check for table indicators: "Example", "Structure", "IC50" etc.
            has_example = False
            has_structure = False
            has_assay = False

            for w in words:
                text_lower = w['text'].lower()
                corrected = self._correct_ocr_assay_name(w['text']).lower()

                if text_lower in ('example', 'synthetic'):
                    has_example = True
                if text_lower == 'structure':
                    has_structure = True
                # Look for IC50/ICso/ICs0/etc. header (may have parentheses)
                corrected_bare = re.sub(r'[()]', '', corrected)
                if corrected_bare in ('ic50', 'icso', 'ics0') or 'ic50' in corrected:
                    has_assay = True
                    assay_header_x = w['x']
                    assay_name = "IC50"
                elif corrected_bare in ('ec50', 'ecso', 'ecs0') or 'ec50' in corrected:
                    has_assay = True
                    assay_header_x = w['x']
                    assay_name = "EC50"
                elif corrected_bare in ('gi50', 'giso', 'gis0') or 'gi50' in corrected:
                    has_assay = True
                    assay_header_x = w['x']
                    assay_name = "GI50"

            # Need example + assay indicator (structure alone is not enough)
            if not has_assay:
                self._debug_info.append(
                    f"  Patent table columns: page {page_num} no assay header found "
                    f"(example={has_example}, structure={has_structure}, assay={has_assay})"
                )
                return data

            self._debug_info.append(
                f"  Patent table columns: page {page_num} is table page "
                f"(assay_header_x={assay_header_x}, assay={assay_name})"
            )

            # Extract example numbers: digits in left margin (x < 28% of width)
            # Skip the top 12% of the page (header area with page numbers)
            x_left_threshold = int(width * 0.28)
            y_header_threshold = int(height * 0.12)
            # Handles: "7", "100", "100-7", "100-11", "12a"
            example_pattern = re.compile(r'^(\d{1,4}(?:-\d{1,4})?[a-zA-Z]?)$')
            examples = []  # [(number_str, y_center)]

            for w in words:
                if w['x'] > x_left_threshold or w['conf'] < 30:
                    continue
                if w['y'] < y_header_threshold:
                    continue  # Skip page header numbers

                text = w['text']

                # Filter out NMR notation (1H, 2H, 3H, 4H)
                if re.match(r'^\d[HhDd]$', text):
                    continue

                # First try exact match (no correction needed)
                match = example_pattern.match(text)

                # If no match, try OCR corrections for short texts only
                # (avoid correcting words like "In" → "1n")
                if not match and len(text) <= 6:
                    corrected_text = text
                    # Only replace i/l/I with 1 if adjacent to a digit
                    corrected_text = re.sub(r'(?<=\d)[ilI]', '1', corrected_text)
                    corrected_text = re.sub(r'[ilI](?=\d)', '1', corrected_text)
                    # Also handle standalone "il" → "11" (two chars, both look like 1)
                    if re.match(r'^[ilI]{1,4}$', text):
                        corrected_text = re.sub(r'[ilI]', '1', text)
                    corrected_text = re.sub(r'[O]', '0', corrected_text)
                    match = example_pattern.match(corrected_text)
                if match:
                    y_center = w['y'] + w['h'] // 2
                    examples.append((self._fix_ocr_compound_id(match.group(1)), y_center))

            # Fix truncated compound IDs: "00-1" → "100-1" when "100" exists
            base_ids = [num for num, _ in examples if '-' not in num and len(num) >= 2]
            if base_ids:
                longest_base = max(base_ids, key=len)  # e.g., "100"
                fixed_examples = []
                for num, y in examples:
                    if '-' in num:
                        prefix = num.split('-', 1)[0]
                        suffix = num.split('-', 1)[1]
                        if len(prefix) < len(longest_base) and longest_base.endswith(prefix):
                            old_num = num
                            num = longest_base + '-' + suffix
                            self._debug_info.append(f"  Fixed truncated ID: {old_num} -> {num}")
                    fixed_examples.append((num, y))
                examples = fixed_examples

            # Filter outlier IDs: data values misidentified as example numbers
            # If most IDs are in range [1-N], reject IDs that are >> N
            examples = self._filter_outlier_compound_ids(examples)

            self._debug_info.append(
                f"  Found {len(examples)} example numbers: {[e[0] for e in examples]}"
            )

            if not examples:
                return data

            # Extract IC50/activity values near the assay column position
            # If we know the header X, look within ±100px of it
            # Otherwise, look in the right 50% of the page
            if assay_header_x is not None:
                val_x_min = assay_header_x - 50
                val_x_max = assay_header_x + 150
            else:
                val_x_min = int(width * 0.50)
                val_x_max = width

            values = []  # [(value_str, y_center)]

            for w in words:
                if w['x'] < val_x_min or w['x'] > val_x_max:
                    continue
                if w['y'] < y_header_threshold:
                    continue  # Skip page header area

                text = w['text']

                # Normalize + symbol OCR misreads
                text = self._normalize_plus_symbols(text)

                normalized_value = None

                # OCR commonly reads "+" as "t", "++" as "tt", "+++" as "ttt"
                # Only in the assay column area (x_pct > 40%)
                if re.match(r'^[tT]+$', text) and w['x_pct'] > 40:
                    t_count = len(text)
                    normalized_value = '+' * t_count

                # Mixed t/+ patterns (e.g., "t+" or "+t")
                elif re.match(r'^[tT+＋]+$', text) and w['x_pct'] > 40 and len(text) <= 5:
                    plus_count = sum(1 for c in text if c in 'tT+＋')
                    normalized_value = '+' * plus_count

                # Mixed +/4 patterns (e.g., "+444" → "++++")
                elif re.match(r'^[+＋4]+$', text) and len(text) >= 2:
                    normalized_value = '+' * len(text)

                # Plus symbols
                elif re.match(r'^[+＋]{1,5}$', text):
                    normalized_value = '+' * (text.count('+') + text.count('＋'))

                # Dash/minus (inactive)
                elif re.match(r'^[-−–—]{1,5}$', text):
                    normalized_value = '-'

                # Numeric values with optional units (but NOT 4-digit page numbers)
                elif re.match(r'^[<>≤≥]?\s*\d{1,3}\.?\d*\s*(?:nM|μM|uM|mM|pM|%)?$', text, re.IGNORECASE):
                    normalized_value = text.strip()

                # ND/NA/NT
                elif text.upper() in ('ND', 'NA', 'NT'):
                    normalized_value = text.upper()

                # "He", "H+" etc. are OCR misreads of "++"
                elif re.match(r'^[Hh][eE+]+$', text) and w['x_pct'] > 40:
                    # "He" → "++", "H+" → "++"
                    normalized_value = '++'

                if normalized_value:
                    y_center = w['y'] + w['h'] // 2
                    values.append((normalized_value, y_center))

            self._debug_info.append(
                f"  Found {len(values)} assay values: {[v[0] for v in values]}"
            )

            if not values:
                return data

            # Match example numbers to values by Y-position proximity
            y_tolerance = 100  # pixels at 200dpi (~1/3 inch)
            used_values = set()

            for ex_num, ex_y in examples:
                best_val = None
                best_dist = float('inf')
                best_idx = -1

                for val_idx, (val_str, val_y) in enumerate(values):
                    if val_idx in used_values:
                        continue
                    dist = abs(ex_y - val_y)
                    if dist < best_dist and dist < y_tolerance:
                        best_dist = dist
                        best_val = val_str
                        best_idx = val_idx

                if best_val is not None:
                    used_values.add(best_idx)
                    bio = BiologicalData(compound_id=ex_num)
                    bio.other_assays[assay_name] = best_val
                    if 'ic50' in assay_name.lower():
                        bio.ic50 = best_val
                    elif 'ec50' in assay_name.lower():
                        bio.ec50 = best_val
                    data[ex_num] = bio
                    self._debug_info.append(
                        f"  Matched: Example {ex_num} (y={ex_y}) -> "
                        f"{assay_name}={best_val} (dist={best_dist:.0f})"
                    )

            self._debug_info.append(
                f"  Patent table columns: extracted {len(data)} compounds on page {page_num}"
            )

        except Exception as e:
            self._debug_info.append(f"  Patent table columns error: {e}")

        return data

    def _extract_from_page_ocr(self, pdfium_page, page_num: int) -> Dict[str, BiologicalData]:
        """Extract biological data using OCR on page image."""
        data = {}

        if not HAS_OCR:
            return data

        try:
            # Render page as image at 200 DPI for reliable OCR
            # (2x was too low for many patents with image-based tables)
            scale = 200.0 / 72.0  # ~2.78x
            bitmap = pdfium_page.render(scale=scale)
            pil_image = bitmap.to_pil()

            width, height = pil_image.size
            self._debug_info.append(f"  Page image size: {width}x{height}")

            # Try targeted patent table column extraction first
            # This handles the common patent layout: Example | Structure (image) | IC50
            self._debug_info.append(f"  Trying patent table column extraction on page {page_num}...")
            try:
                data = self._extract_patent_table_columns(pil_image, page_num)
                if data:
                    self._debug_info.append(f"  Patent table columns found {len(data)} compounds")
                    return data
            except Exception as e:
                self._debug_info.append(f"  Patent table columns failed: {e}")

            # Try structured OCR (preserves table layout)
            self._debug_info.append(f"  Running structured OCR on page {page_num}...")
            try:
                data = self._extract_table_from_ocr(pil_image, page_num)
                if data:
                    self._debug_info.append(f"  Structured OCR found {len(data)} compounds")
                    return data
            except Exception as e:
                self._debug_info.append(f"  Structured OCR failed: {e}")

            # Fallback: Run OCR on full page
            self._debug_info.append(f"  Running text OCR on full page {page_num}...")
            ocr_text_full = pytesseract.image_to_string(pil_image)

            # Skip NMR/analytical pages in the OCR fallback too
            if self._is_nmr_analytical_page(ocr_text_full):
                self._debug_info.append(f"  Skipping NMR/analytical page (OCR fallback)")
                return data

            # Also try splitting into left and right halves for two-column layouts
            self._debug_info.append(f"  Running OCR on left/right halves...")

            # Left half
            left_half = pil_image.crop((0, 0, width // 2, height))
            ocr_text_left = pytesseract.image_to_string(left_half)

            # Right half
            right_half = pil_image.crop((width // 2, 0, width, height))
            ocr_text_right = pytesseract.image_to_string(right_half)

            # Combine all OCR text
            all_ocr_text = ocr_text_full + "\n" + ocr_text_left + "\n" + ocr_text_right

            self._debug_info.append(f"  Full page OCR: {len(ocr_text_full)} chars")
            self._debug_info.append(f"  Left half OCR: {len(ocr_text_left)} chars")
            self._debug_info.append(f"  Right half OCR: {len(ocr_text_right)} chars")

            # Show samples
            sample_left = ocr_text_left[:300].replace('\n', ' | ')
            sample_right = ocr_text_right[:300].replace('\n', ' | ')
            self._debug_info.append(f"  Left sample: {sample_left}...")
            self._debug_info.append(f"  Right sample: {sample_right}...")

            # Parse all OCR text for biological data
            data = self._parse_bio_text(all_ocr_text, page_num)

        except Exception as e:
            self._debug_info.append(f"  OCR extraction error: {e}")

        return data

    def _extract_table_from_ocr(self, pil_image, page_num: int) -> Dict[str, BiologicalData]:
        """Extract table data using OCR with layout preservation."""
        data = {}

        # Check for two-column table layout by looking at full page OCR first
        full_text = pytesseract.image_to_string(pil_image)

        # Skip NMR/analytical pages entirely
        if self._is_nmr_analytical_page(full_text):
            self._debug_info.append(f"  Skipping NMR/analytical page in structured OCR")
            return data

        # Detect two-column bio data table
        is_two_column = self._detect_two_column_layout(full_text, pil_image)

        if is_two_column:
            self._debug_info.append(f"  Detected two-column table layout on page {page_num}")
            width, height = pil_image.size

            # Process each half with the full structured OCR pipeline
            left_half = pil_image.crop((0, 0, width // 2, height))
            right_half = pil_image.crop((width // 2, 0, width, height))

            left_data = self._extract_two_col_half(left_half, page_num, "left")
            data.update(left_data)

            right_data = self._extract_two_col_half(right_half, page_num, "right")
            # Merge right data, preferring existing left data for same compound IDs
            for cid, bio in right_data.items():
                if cid in data:
                    self._merge_bio_data(data[cid], bio)
                else:
                    data[cid] = bio

            if data:
                return data

        # Get OCR data with bounding boxes (as dictionary, no pandas needed)
        ocr_dict = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)

        # Build list of word entries with position and dimensions
        # Filter out table border characters and garbage
        words = []
        n_boxes = len(ocr_dict['text'])
        for i in range(n_boxes):
            text = str(ocr_dict['text'][i]).strip()
            conf = int(ocr_dict['conf'][i]) if ocr_dict['conf'][i] != '-1' else 0

            # Filter empty, low confidence, and table border characters
            if not text or conf < 30:
                continue
            # Skip table borders and common OCR garbage
            if text in ('|', '||', '|||', '[', ']', '{', '}', '(', ')', '_', '__'):
                continue
            # Skip if mostly non-alphanumeric (likely OCR garbage)
            alphanumeric = sum(1 for c in text if c.isalnum() or c in '.<>+-')
            if len(text) > 1 and alphanumeric < len(text) * 0.3:
                continue

            words.append({
                'text': text,
                'left': ocr_dict['left'][i],
                'top': ocr_dict['top'][i],
                'width': ocr_dict['width'][i],
                'height': ocr_dict['height'][i],
                'conf': conf
            })

        if not words:
            return data

        # Sort by vertical position then horizontal
        words.sort(key=lambda w: (w['top'], w['left']))

        # Cluster into rows (words within 15 pixels vertically are same row)
        row_threshold = 15
        rows = []
        current_row = []
        current_top = words[0]['top'] if words else 0

        for word in words:
            if abs(word['top'] - current_top) > row_threshold:
                # New row
                if current_row:
                    current_row.sort(key=lambda w: w['left'])
                    # Clean cells - remove remaining "|" and other border artifacts
                    row_cells = []
                    for w in current_row:
                        clean_text = w['text'].strip('|').strip()
                        if clean_text:
                            row_cells.append(clean_text)
                    row_text = ' '.join(row_cells)
                    row_positions = [w['left'] for w in current_row if w['text'].strip('|').strip()]
                    if row_cells:  # Only add non-empty rows
                        rows.append({'text': row_text, 'cells': row_cells, 'positions': row_positions, 'words': current_row})
                current_row = [word]
                current_top = word['top']
            else:
                current_row.append(word)

        # Don't forget last row
        if current_row:
            current_row.sort(key=lambda w: w['left'])
            row_cells = []
            for w in current_row:
                clean_text = w['text'].strip('|').strip()
                if clean_text:
                    row_cells.append(clean_text)
            row_text = ' '.join(row_cells)
            row_positions = [w['left'] for w in current_row if w['text'].strip('|').strip()]
            if row_cells:
                rows.append({'text': row_text, 'cells': row_cells, 'positions': row_positions, 'words': current_row})

        self._debug_info.append(f"  OCR found {len(rows)} text rows")
        if rows:
            self._debug_info.append(f"  First few rows: {[r['text'][:80] for r in rows[:5]]}")

        # Find header row using improved detection
        header_idx, header_columns, header_is_bio = self._find_table_header_ocr(rows, page_num)

        # If header found but it's a structure-only table, skip entire page extraction
        if header_idx is not None and not header_is_bio:
            self._debug_info.append(f"  Skipping page - header is structure-related, not bio data")
            return data

        # If no header found on this page, use cached header from previous pages
        if header_idx is None and self._cached_header_columns:
            header_columns = self._cached_header_columns
            self._debug_info.append(f"  Using cached header from previous page: {header_columns[:5]}...")

        # Start parsing from after header, or from beginning if no header
        start_idx = header_idx + 1 if header_idx is not None else 0

        if header_idx is None and not header_columns:
            self._debug_info.append(f"  No table header found - will try pattern-based extraction")

        # Parse data rows - look for rows with compound IDs and numeric values
        effective_headers = header_columns if header_columns else self._cached_header_columns

        # Collect data rows with and without compound IDs
        rows_with_ids = []  # [(compound_id, values)]
        rows_without_ids = []  # [values] - for positional matching later

        for row_idx, row in enumerate(rows[start_idx:]):
            cells = row['cells']
            if not cells:
                continue

            # Skip rows that look like headers or document structure
            row_text = row['text'].lower()
            if 'pct' in row_text and 'wo' in row_text:  # Skip page headers
                continue

            # Skip rows that look like OCR garbage from chemical structures
            # These often have: single letters, symbols like = \ / z, very short cells
            garbage_indicators = 0
            for cell in cells:
                cell_clean = cell.strip()
                # Single non-digit character (likely structure artifact)
                if len(cell_clean) == 1 and not cell_clean.isdigit():
                    garbage_indicators += 1
                # Common OCR artifacts from structures
                if cell_clean in ('z', 'Z', '\\', '/', '=', '~', 'x', 'X', 'woo', 'fe}', 'ra', 'uu'):
                    garbage_indicators += 1
                # Looks like chemical notation (N, O, F, etc. with no context)
                if re.match(r'^[NOFHCS]$', cell_clean):
                    garbage_indicators += 1

            # If >50% of cells look like garbage, skip this row
            if len(cells) > 0 and garbage_indicators / len(cells) > 0.5:
                continue

            # Try to find compound ID - could be in first cell or among first few cells
            compound_id = None
            data_start_idx = 0

            for cell_idx, cell in enumerate(cells[:2]):  # Check first 2 cells only
                cell_clean = cell.strip()
                # Look for compound ID patterns (1, 2a, 10, 99, etc.)
                # Also handle OCR errors like "[1]" or "(1)"
                cell_for_match = re.sub(r'[\[\]\(\)\{\}]', '', cell_clean)
                # Compound IDs: 1-9999, with optional dash suffix or letter
                # e.g., "7", "100", "100-7", "12a"
                compound_match = re.match(r'^(\d{1,4}(?:-\d{1,4})?[a-zA-Z]?)$', cell_for_match)
                if compound_match:
                    cid_candidate = self._fix_ocr_compound_id(compound_match.group(1))
                    # Filter out NMR notation (1H, 2H, 3H, 4H)
                    if re.match(r'^\d[HhDd]$', cid_candidate):
                        continue
                    compound_id = cid_candidate
                    data_start_idx = cell_idx + 1
                    break

            # Check if this row has numeric values (indicates a data row)
            data_cells = cells[data_start_idx:]
            numeric_values = []
            for cell in data_cells:
                cell_clean = cell.strip()
                # Skip obvious garbage
                if cell_clean in ('|', '||', '[', ']', '{', '}', '(', ')'):
                    numeric_values.append('')
                    continue
                # Clean up cell - remove stray | characters
                cell_clean = cell_clean.strip('|').strip()
                if not cell_clean:
                    numeric_values.append('')
                    continue
                # Check for numeric value patterns
                if re.search(r'^[<>≤≥±~]?\s*-?\d+\.?\d*$', cell_clean):
                    numeric_values.append(cell_clean)
                elif re.search(r'^[<>]\s*\d+', cell_clean):  # >1000, <10
                    numeric_values.append(cell_clean)
                elif cell_clean.lower() in ('+', '++', '+++', '++++', '+++++', '-', '--', '---', 'nd', 'na', 'nt', 'inactive', 'active'):
                    numeric_values.append(cell_clean)
                elif re.search(r'^\d+\.?\d*\s*n[mM]$', cell_clean):  # Value with unit
                    numeric_values.append(cell_clean)
                elif re.search(r'^\d+\.?\d*\s*%$', cell_clean):  # Percentage
                    numeric_values.append(cell_clean)
                else:
                    numeric_values.append('')  # Keep position

            # Consider rows as data if they have numeric values
            # Rows WITH compound IDs need only 1 numeric value (e.g., "100  compound_name  0.173")
            # Rows WITHOUT IDs need at least 2 (to avoid false positives)
            non_empty_values = [v for v in numeric_values if v]
            min_values = 1 if compound_id else 2
            if len(non_empty_values) >= min_values:
                if compound_id:
                    rows_with_ids.append((compound_id, numeric_values))
                    self._debug_info.append(f"  Data row with ID {compound_id}: {numeric_values[:5]}...")
                else:
                    rows_without_ids.append(numeric_values)
                    self._debug_info.append(f"  Data row without ID (idx {len(rows_without_ids)}): {numeric_values[:5]}...")

        # Post-process: recover truncated compound ID prefixes
        # e.g., if we have "100" and "00-1", "00-2", the "00-X" should be "100-X"
        if rows_with_ids:
            all_ids = [cid for cid, _ in rows_with_ids]
            # Find potential prefix: IDs without dashes that are longer
            base_ids = [cid for cid in all_ids if '-' not in cid and len(cid) >= 2]
            dash_ids = [cid for cid in all_ids if '-' in cid]

            if base_ids and dash_ids:
                # Check if dash IDs look like they're missing a prefix digit
                common_prefix = base_ids[0]  # e.g., "100"
                fixed_rows = []
                for compound_id, values in rows_with_ids:
                    if '-' in compound_id:
                        parts = compound_id.split('-', 1)
                        # If the prefix part is shorter than expected and looks truncated
                        # e.g., "00-1" when we expect "100-1"
                        if len(parts[0]) < len(common_prefix) and common_prefix.endswith(parts[0]):
                            old_id = compound_id
                            compound_id = common_prefix + '-' + parts[1]
                            self._debug_info.append(f"  Fixed truncated ID: {old_id} -> {compound_id}")
                    fixed_rows.append((compound_id, values))
                rows_with_ids = fixed_rows

        # Filter outlier compound IDs (data values misidentified as IDs)
        if len(rows_with_ids) >= 5:
            id_tuples = [(cid, 0) for cid, _ in rows_with_ids]
            filtered_tuples = self._filter_outlier_compound_ids(id_tuples)
            filtered_ids = {cid for cid, _ in filtered_tuples}
            rows_with_ids = [(cid, vals) for cid, vals in rows_with_ids if cid in filtered_ids]

        # Check if this header is structure-related or NMR/analytical (not bio data)
        header_is_structure_only = False
        if effective_headers:
            header_text = ' '.join(effective_headers).lower()
            bio_kws = ['ic50', 'ec50', 'ki', 'kd', 'gi50', 'emax', 'inhibition',
                       'activity', 'competition', 'binding', 'atpase', 'potency',
                       'assay 1', 'assay 2', 'assay 3', 'antagonist']
            struct_kws = ['structure', 'formula', 'smiles']
            nmr_kws = ['nmr', 'ppm', 'mhz', 'lcms', 'hplc', 'analytical', 'iupac',
                       'physicochemical', 'optical rotation']
            has_bio = any(kw in header_text for kw in bio_kws)
            has_struct = any(kw in header_text for kw in struct_kws)
            has_nmr = any(kw in header_text for kw in nmr_kws)
            if (has_struct or has_nmr) and not has_bio:
                header_is_structure_only = True
                self._debug_info.append(f"  Skipping structure/analytical table: {header_text[:60]}")

        # Build results from rows with IDs (skip structure-only tables)
        if not header_is_structure_only:
            for compound_id, values in rows_with_ids:
                bio_data = BiologicalData(compound_id=compound_id)
                for i, value in enumerate(values):
                    if value:
                        if effective_headers and i < len(effective_headers):
                            col_name = effective_headers[i]
                        else:
                            col_name = f"Assay_{i + 1}"
                        bio_data.other_assays[col_name] = value
                if bio_data.other_assays:
                    data[compound_id] = bio_data

        # If we have rows without IDs but no rows with IDs, consider positional matching
        # But only if we have a valid bio data header (not structure-related)
        if rows_without_ids and not rows_with_ids:
            # Check if header looks like actual bio data (not "Cpd. Structure" etc.)
            header_is_bio_data = False
            if effective_headers:
                header_text = ' '.join(effective_headers).lower()
                # Bio data headers contain assay keywords
                bio_keywords = ['ic50', 'ec50', 'ki', 'kd', 'gi50', 'emax', 'inhibition',
                               'activity', 'competition', 'binding', 'nm', 'μm', '%']
                header_is_bio_data = any(kw in header_text for kw in bio_keywords)
                # Exclude structure-related headers
                structure_keywords = ['structure', 'cpd.', 'compound', 'formula', 'name']
                if any(kw in header_text for kw in structure_keywords) and not any(kw in header_text for kw in bio_keywords[:6]):
                    header_is_bio_data = False

            # Also check if values look like real bio data (not just single digits)
            values_look_like_bio_data = False
            for row_values in rows_without_ids[:3]:  # Check first 3 rows
                for val in row_values:
                    if val:
                        # Real bio data: +/++/+++, >1000, <10, values with units, negative numbers
                        if re.search(r'^[+]{2,}$|^[<>]\d+|^\d+\.?\d*\s*[nμm]M|^-\d+', val):
                            values_look_like_bio_data = True
                            break
                        # Also accept larger numbers (>10) as likely bio data
                        if re.search(r'^\d{2,}\.?\d*$', val):  # 2+ digit numbers
                            values_look_like_bio_data = True
                            break
                if values_look_like_bio_data:
                    break

            if header_is_bio_data or values_look_like_bio_data:
                self._debug_info.append(f"  Using positional matching for {len(rows_without_ids)} data rows")
                self._debug_info.append(f"    header_is_bio_data={header_is_bio_data}, values_look_like_bio_data={values_look_like_bio_data}")
                # Track next compound ID to assign (continue from previous pages)
                next_id = getattr(self, '_next_positional_id', 1)

                for row_values in rows_without_ids:
                    compound_id = str(next_id)
                    bio_data = BiologicalData(compound_id=compound_id)
                    for i, value in enumerate(row_values):
                        if value:
                            if effective_headers and i < len(effective_headers):
                                col_name = effective_headers[i]
                            else:
                                col_name = f"Assay_{i + 1}"
                            bio_data.other_assays[col_name] = value
                    if bio_data.other_assays:
                        data[compound_id] = bio_data
                        self._debug_info.append(f"  Positional match: Compound {compound_id} -> {list(bio_data.other_assays.items())[:3]}")
                    next_id += 1

                self._next_positional_id = next_id
            else:
                self._debug_info.append(f"  Skipping positional matching - header doesn't look like bio data")

        return data

    def _extract_single_column_table(
        self, pil_image, page_num: int, side: str
    ) -> Dict[str, BiologicalData]:
        """Extract bio data from a single column of a two-column table.

        Args:
            pil_image: PIL Image of one half of the page.
            page_num: Page number for debugging.
            side: "left" or "right" for debugging.

        Returns:
            Dictionary mapping compound_id to BiologicalData.
        """
        data = {}

        try:
            # Run OCR on this half
            ocr_text = pytesseract.image_to_string(pil_image)
            lines = ocr_text.strip().split('\n')

            self._debug_info.append(f"  {side.capitalize()} half: {len(lines)} lines")

            # Find header line with assay info (IC50, EC50, etc.)
            header_info = None
            header_idx = -1
            for idx, line in enumerate(lines):
                line_lower = line.lower()
                # Look for assay keywords
                if any(kw in line_lower for kw in ['ic50', 'ic 50', 'ec50', 'ec 50', 'competition', 'inhibition', 'activity']):
                    header_info = line
                    header_idx = idx
                    self._debug_info.append(f"  {side.capitalize()} header at line {idx}: {line[:60]}...")
                    break

            if header_idx < 0:
                return data

            # Parse data lines after header
            # Pattern: compound_number followed by value(s)
            compound_pattern = re.compile(r'^\s*(\d{1,4}(?:-\d{1,4})?[a-zA-Z]?)\s+(.+)$')
            value_pattern = re.compile(r'[+＋]{1,5}|[-−–—]{1,5}|[<>]?\s*\d+\.?\d*\s*(?:nM|μM|%)?|ND|NA|NT', re.IGNORECASE)

            for line in lines[header_idx + 1:]:
                line = line.strip()
                if not line:
                    continue

                # Skip page headers/footers
                if 'WO 20' in line or 'PCT/' in line:
                    continue

                match = compound_pattern.match(line)
                if match:
                    compound_id = self._fix_ocr_compound_id(match.group(1))
                    rest = match.group(2)

                    # Extract values from the rest of the line
                    values = value_pattern.findall(rest)
                    if values:
                        bio_data = BiologicalData(compound_id=compound_id)

                        # Use header info to name the assay
                        assay_name = "IC50"  # Default
                        if header_info:
                            # Apply OCR correction to header
                            corrected_header = self._correct_ocr_assay_name(header_info)
                            # Try to extract assay name from header
                            if 'competition' in corrected_header.lower():
                                assay_name = "Competition IC50"
                            elif 'ic50' in corrected_header.lower() or 'ic 50' in corrected_header.lower():
                                assay_name = "IC50"
                            elif 'ec50' in corrected_header.lower():
                                assay_name = "EC50"
                            else:
                                # Use cleaned and corrected header as assay name
                                assay_name = self._correct_ocr_assay_name(' '.join(header_info.split()[:3]))

                        # Store primary value
                        bio_data.other_assays[assay_name] = values[0]

                        # Also set legacy fields if applicable
                        if 'ic50' in assay_name.lower():
                            bio_data.ic50 = values[0]
                        elif 'ec50' in assay_name.lower():
                            bio_data.ec50 = values[0]

                        data[compound_id] = bio_data
                        self._debug_info.append(f"  {side.capitalize()}: Compound {compound_id} -> {assay_name}={values[0]}")

        except Exception as e:
            self._debug_info.append(f"  {side.capitalize()} extraction error: {e}")

        return data

    def _detect_two_column_layout(self, full_text: str, pil_image) -> bool:
        """Detect if a page has a two-column bio data table layout.

        Checks for patterns like:
        - Repeated headers (Assay 1...Assay 1, IC50...IC50, EC50...EC50)
        - "TABLE X" and "TABLE X-continued" side by side
        - "Exam-" appearing twice in the same line (line-wrapped "Example")
        - Two "No." or "Ex." columns detected via OCR position analysis
        """
        # Text-based detection patterns
        two_col_patterns = [
            r'(?:Compound|Cpd)\s+(?:#|No).*(?:Compound|Cpd)\s+(?:#|No)',
            r'IC\s*50.*IC\s*50',
            r'EC\s*50.*EC\s*50',
            r'Assay\s+\d.*Assay\s+\d.*Assay\s+\d.*Assay\s+\d',  # Multiple assay columns repeated
            r'Exam-.*Exam-',  # Line-wrapped "Example" twice
            r'TABLE\s+\d+\s+TABLE\s+\d+',  # Two TABLE headers side by side
            r'TABLE\s+\d+-continued\s+TABLE\s+\d+',  # TABLE continued + TABLE
            r'K[,;i]\s*.*K[,;i]\s*.*K[,;i]\s*.*K[,;i]',  # Multiple Ki columns (KAT6 style)
        ]
        if any(re.search(p, full_text, re.IGNORECASE) for p in two_col_patterns):
            return True

        # Position-based detection: use OCR data to check if "Example" or "No."
        # appears on both left and right halves
        try:
            width = pil_image.size[0]
            ocr_data = pytesseract.image_to_data(
                pil_image, output_type=pytesseract.Output.DICT
            )
            example_positions = []
            for i in range(len(ocr_data['text'])):
                text = str(ocr_data['text'][i]).strip().lower()
                if text in ('example', 'exam-', 'ex.', 'no.'):
                    x = ocr_data['left'][i]
                    example_positions.append(x)
            # If "Example"/"No." appears on both halves, it's two-column
            if example_positions:
                left_count = sum(1 for x in example_positions if x < width * 0.4)
                right_count = sum(1 for x in example_positions if x > width * 0.5)
                if left_count >= 1 and right_count >= 1:
                    return True
        except Exception:
            pass

        return False

    def _extract_two_col_half(self, pil_image, page_num: int, side: str) -> Dict[str, BiologicalData]:
        """Extract bio data from one half of a two-column table.

        Uses the full structured OCR pipeline (header detection, row parsing)
        on the cropped half-page image.
        """
        data = {}

        try:
            # Get OCR data with bounding boxes
            ocr_dict = pytesseract.image_to_data(
                pil_image, output_type=pytesseract.Output.DICT
            )

            # Build word entries
            words = []
            n_boxes = len(ocr_dict['text'])
            for i in range(n_boxes):
                text = str(ocr_dict['text'][i]).strip()
                conf = int(ocr_dict['conf'][i]) if ocr_dict['conf'][i] != '-1' else 0
                if not text or conf < 30:
                    continue
                if text in ('|', '||', '|||', '[', ']', '{', '}', '(', ')', '_', '__'):
                    continue
                alphanumeric = sum(1 for c in text if c.isalnum() or c in '.<>+-')
                if len(text) > 1 and alphanumeric < len(text) * 0.3:
                    continue
                words.append({
                    'text': text,
                    'left': ocr_dict['left'][i],
                    'top': ocr_dict['top'][i],
                    'width': ocr_dict['width'][i],
                    'height': ocr_dict['height'][i],
                    'conf': conf
                })

            if not words:
                return data

            # Sort and cluster into rows
            words.sort(key=lambda w: (w['top'], w['left']))
            row_threshold = 15
            rows = []
            current_row = []
            current_top = words[0]['top']

            for word in words:
                if abs(word['top'] - current_top) > row_threshold:
                    if current_row:
                        current_row.sort(key=lambda w: w['left'])
                        row_cells = [w['text'].strip('|').strip() for w in current_row]
                        row_cells = [c for c in row_cells if c]
                        row_text = ' '.join(row_cells)
                        row_positions = [w['left'] for w in current_row if w['text'].strip('|').strip()]
                        if row_cells:
                            rows.append({'text': row_text, 'cells': row_cells, 'positions': row_positions, 'words': current_row})
                    current_row = [word]
                    current_top = word['top']
                else:
                    current_row.append(word)

            if current_row:
                current_row.sort(key=lambda w: w['left'])
                row_cells = [w['text'].strip('|').strip() for w in current_row]
                row_cells = [c for c in row_cells if c]
                row_text = ' '.join(row_cells)
                row_positions = [w['left'] for w in current_row if w['text'].strip('|').strip()]
                if row_cells:
                    rows.append({'text': row_text, 'cells': row_cells, 'positions': row_positions, 'words': current_row})

            self._debug_info.append(f"  {side.capitalize()} half: {len(rows)} OCR rows")

            # Find header in this half
            header_idx, header_columns, header_is_bio = self._find_table_header_ocr(rows, page_num)

            # Skip structure-only headers
            if header_idx is not None and not header_is_bio:
                header_idx = None
                header_columns = []

            # Use cached header if none found
            effective_headers = header_columns if header_columns else self._cached_header_columns

            start_idx = header_idx + 1 if header_idx is not None else 0

            # Parse data rows
            compound_pattern = re.compile(r'^(\d{1,4}(?:-\d{1,4})?[a-zA-Z]?)$')

            # Determine the leftmost column X position for compound IDs
            # Only accept IDs from words at the expected X position
            id_x_positions = []
            for row in rows[start_idx:]:
                if row.get('words'):
                    first_word = row['words'][0]
                    text = first_word['text'].strip()
                    cell_clean = re.sub(r'[\[\]\(\)\{\}]', '', text)
                    if compound_pattern.match(cell_clean):
                        id_x_positions.append(first_word['left'])
            # The typical compound ID X position (most common leftmost X)
            id_x_max = None
            if id_x_positions:
                # Use the 25th percentile as max allowed X for compound IDs
                id_x_positions.sort()
                id_x_max = id_x_positions[min(len(id_x_positions) // 4 + 1, len(id_x_positions) - 1)] + 30

            collected_rows = []  # (compound_id, values)

            for row in rows[start_idx:]:
                cells = row['cells']
                if not cells:
                    continue

                # Skip garbage rows
                row_text = row['text'].lower()
                if 'pct' in row_text and 'wo' in row_text:
                    continue
                if 'us ' in row_text and re.search(r'us\s+\d{4}', row_text):
                    continue

                # Find compound ID - only from first cell at the expected X position
                compound_id = None
                data_start = 0
                for ci, cell in enumerate(cells[:2]):
                    cell_clean = re.sub(r'[\[\]\(\)\{\}]', '', cell.strip())
                    m = compound_pattern.match(cell_clean)
                    if m:
                        # Filter NMR notation
                        if re.match(r'^\d[HhDd]$', m.group(1)):
                            continue
                        # Check X position if available
                        if id_x_max and row.get('words') and ci < len(row['words']):
                            word_x = row['words'][ci]['left']
                            if word_x > id_x_max:
                                continue  # This number is in a data column, not ID column
                        compound_id = self._fix_ocr_compound_id(m.group(1))
                        data_start = ci + 1
                        break

                if not compound_id:
                    continue

                # Extract numeric values
                data_cells = cells[data_start:]
                values = []
                for cell in data_cells:
                    cell_clean = cell.strip().strip('|').strip()
                    if not cell_clean:
                        values.append('')
                    elif re.search(r'^[<>≤≥±~]?\s*-?\d+\.?\d*$', cell_clean):
                        values.append(cell_clean)
                    elif re.search(r'^[<>]\s*\d+', cell_clean):
                        values.append(cell_clean)
                    elif cell_clean.lower() in ('+', '++', '+++', '++++', '-', 'nd', 'na', 'nt', 'n/d'):
                        values.append(cell_clean)
                    elif re.search(r'^\d+\.?\d*\s*(?:nM|μM|%|uM)$', cell_clean, re.IGNORECASE):
                        values.append(cell_clean)
                    else:
                        values.append('')

                non_empty = [v for v in values if v]
                if non_empty:
                    bio_data = BiologicalData(compound_id=compound_id)
                    for i, value in enumerate(values):
                        if value:
                            if effective_headers and i < len(effective_headers):
                                col_name = effective_headers[i]
                            else:
                                col_name = f"Assay_{i + 1}"
                            bio_data.other_assays[col_name] = value
                    if bio_data.other_assays:
                        data[compound_id] = bio_data
                        self._debug_info.append(f"  {side.capitalize()}: Compound {compound_id} -> {list(bio_data.other_assays.items())[:3]}")

            # Apply outlier compound ID filter
            if len(data) >= 5:
                examples = [(cid, 0) for cid in data.keys()]
                filtered = self._filter_outlier_compound_ids(examples)
                filtered_ids = {cid for cid, _ in filtered}
                removed = set(data.keys()) - filtered_ids
                for cid in removed:
                    del data[cid]

        except Exception as e:
            self._debug_info.append(f"  {side.capitalize()} half extraction error: {e}")

        return data

    def _find_table_header_ocr(self, rows: List[dict], page_num: int) -> Tuple[Optional[int], List[str], bool]:
        """Find the actual table header row from OCR'd rows.

        Returns (header_row_index, list_of_column_headers, is_bio_header).
        Distinguishes between actual table headers and legend/explanation text.
        """
        # Patterns that indicate a real table header column
        # These should have assay type + optional units
        header_col_patterns = [
            re.compile(r'(?:GI|IC|EC)[-_]?\s*[5s][0oO]', re.IGNORECASE),  # GI50, IC50, EC50, ICs0, ICso
            re.compile(r'\bICs\b', re.IGNORECASE),  # ICs (OCR misread of IC50)
            re.compile(r'E\s*max', re.IGNORECASE),  # Emax
            re.compile(r'\([nuμ][mM]\)|\(%\)|\(aM\)|\(pM\)', re.IGNORECASE),  # Units: (nM), (μM), (uM), (aM), (%)
            re.compile(r'Ki\b|Kd\b|K[,;]\s*at', re.IGNORECASE),  # Ki, Kd, K;at (OCR of Ki,at)
            re.compile(r'competition|inhibition|activity|binding|ATPase|AcCoA|ACCOA', re.IGNORECASE),
            re.compile(r'Assay\s+\d', re.IGNORECASE),  # Assay 1, Assay 2, etc.
            re.compile(r'antagonist|agonist', re.IGNORECASE),
        ]

        # Patterns that indicate legend/explanation text (NOT a header)
        legend_patterns = [
            re.compile(r'=\s*\w', re.IGNORECASE),  # "= something" (definition)
            re.compile(r'wildtype|wild-type|mutant|resistant', re.IGNORECASE),
            re.compile(r'cancer\s+cells', re.IGNORECASE),
            re.compile(r'derived\s+from', re.IGNORECASE),
            re.compile(r'software|ChemDraw|refers\s+to', re.IGNORECASE),  # Description text
            re.compile(r'embodiment|invention|provides', re.IGNORECASE),  # Patent claim text
            re.compile(r'J\s*=\s*[\d.]+\s*Hz', re.IGNORECASE),  # NMR J-coupling
            re.compile(r'\d+\s*MHz', re.IGNORECASE),  # NMR frequency
            re.compile(r'DMSO|CDCl|methanol-d', re.IGNORECASE),  # NMR solvents
        ]

        # Pattern for "Compound" or "No." column header
        compound_col_pattern = re.compile(r'compound|cpd|cmpd|no\.?|#|example|ex\.?', re.IGNORECASE)

        best_header_idx = None
        best_header_score = 0
        best_header_columns = []

        for idx, row in enumerate(rows):
            row_text = row['text']

            # Skip rows that look like legend text
            if any(p.search(row_text) for p in legend_patterns):
                self._debug_info.append(f"  Row {idx} looks like legend, skipping: {row_text[:60]}...")
                continue

            # Count how many header column patterns match
            header_score = 0
            for pattern in header_col_patterns:
                if pattern.search(row_text):
                    header_score += 1

            # Bonus for having "Compound" or "No." column
            if compound_col_pattern.search(row_text):
                header_score += 0.5

            # Bonus for having multiple distinct "cells" (columns)
            if len(row['cells']) >= 3:
                header_score += 0.5

            # Penalty for very long rows (likely paragraph text, not header)
            if len(row_text) > 150 and len(row['cells']) < 5:
                header_score -= 2

            if header_score > best_header_score:
                best_header_score = header_score
                best_header_idx = idx
                best_header_columns = self._build_header_columns_from_row(row)

        if best_header_idx is not None and best_header_score >= 1:
            self._debug_info.append(f"  Found table header at row {best_header_idx} (score={best_header_score}): {rows[best_header_idx]['text'][:100]}...")

            # Try to merge with subsequent rows that look like header continuations
            # (contain units like (nM), (μM), (%), or short header fragments like "No.", "ple")
            last_header_idx = best_header_idx
            unit_pattern = re.compile(r'\([nuμ][mM]\)|\(%\)|\(uM\)|\(aM\)|\(pM\)', re.IGNORECASE)
            header_frag_pattern = re.compile(r'^(No\.?|ple|Ratio|Number|Emax|count)$', re.IGNORECASE)

            for next_idx in range(best_header_idx + 1, min(best_header_idx + 3, len(rows))):
                next_row = rows[next_idx]
                next_text = next_row['text']
                # Check if this row has units or looks like a header continuation
                has_units = bool(unit_pattern.search(next_text))
                # Check if most cells are short (header fragments, not data)
                cells = next_row['cells']
                short_cells = sum(1 for c in cells if len(c.strip()) <= 8)
                is_mostly_short = len(cells) > 0 and short_cells / len(cells) > 0.6
                # Check for header fragment keywords
                has_header_frags = any(header_frag_pattern.match(c.strip()) for c in cells)

                if has_units or (is_mostly_short and has_header_frags):
                    # Merge this row into the header
                    last_header_idx = next_idx
                    # Append row text to build combined header
                    merged_columns = self._merge_header_rows(
                        rows[best_header_idx:last_header_idx + 1]
                    )
                    if merged_columns:
                        best_header_columns = merged_columns
                    self._debug_info.append(f"  Merged header row {next_idx}: {next_text[:60]}...")
                else:
                    break

            # Only cache headers that look like actual bio data (have assay keywords)
            # Check combined header text from all merged rows
            combined_header_text = ' '.join(rows[i]['text'] for i in range(best_header_idx, last_header_idx + 1)).lower()
            # Apply OCR corrections to header text before keyword matching
            combined_header_text = self._correct_ocr_assay_name(combined_header_text)
            bio_keywords = ['ic50', 'ic 50', 'ic5o', 'ics0', 'icso',
                           'ec50', 'ec 50', 'ec5o', 'ecs0', 'ecso',
                           'ki', 'kd', 'gi50',
                           'emax', 'inhibition', 'activity', 'competition', 'binding',
                           'atpase', 'potency', 'assay 1', 'assay 2', 'antagonist',
                           'accoa', 'acoa']
            structure_only_keywords = ['structure', 'formula', 'name', 'smiles', 'exemplary compounds']
            nmr_keywords = ['nmr', 'ppm', 'mhz', 'lcms', 'hplc', 'analytical',
                           'iupac', 'physicochemical', 'optical rotation']

            is_bio_header = any(kw in combined_header_text for kw in bio_keywords)
            is_structure_only = (any(kw in combined_header_text for kw in structure_only_keywords) or
                               any(kw in combined_header_text for kw in nmr_keywords)) and not is_bio_header

            # Clean up header column names
            best_header_columns = self._clean_header_columns(best_header_columns)

            header_is_bio = is_bio_header and not is_structure_only
            if header_is_bio:
                self._cached_header_columns = best_header_columns
                self._debug_info.append(f"  Cached bio data header: {best_header_columns[:5]}...")
            else:
                self._debug_info.append(f"  Header not cached (structure-related): {rows[best_header_idx]['text'][:60]}...")

            return last_header_idx, best_header_columns, header_is_bio

        return None, [], False

    def _merge_header_rows(self, header_rows: List[dict]) -> List[str]:
        """Merge multiple header rows into a single set of column headers.

        Each header row may have words at various X positions. Words at similar
        X positions across rows are combined into a single column header.
        """
        if len(header_rows) == 1:
            return self._build_header_columns_from_row(header_rows[0])

        # Collect all words from all header rows with their positions
        all_words = []
        for row_idx, row in enumerate(header_rows):
            for word in row.get('words', []):
                all_words.append({
                    'text': word['text'],
                    'left': word['left'],
                    'width': word['width'],
                    'row': row_idx,
                })

        if not all_words:
            # Fallback: just concatenate cells
            combined_cells = []
            for row in header_rows:
                combined_cells.extend(row['cells'])
            return [self._correct_ocr_assay_name(c) for c in combined_cells]

        # Group words by X position into columns
        all_words.sort(key=lambda w: w['left'])
        column_gap = 50
        columns = []
        current_col = []
        last_right = 0

        for word in all_words:
            if current_col and (word['left'] - last_right) > column_gap:
                col_text = ' '.join(w['text'] for w in sorted(current_col, key=lambda w: (w['row'], w['left'])))
                columns.append(col_text)
                current_col = []
            current_col.append(word)
            last_right = max(last_right, word['left'] + word['width'])

        if current_col:
            col_text = ' '.join(w['text'] for w in sorted(current_col, key=lambda w: (w['row'], w['left'])))
            columns.append(col_text)

        return [self._correct_ocr_assay_name(col) for col in columns]

    def _build_header_columns_from_row(self, row: dict) -> List[str]:
        """Build column headers from an OCR'd row, grouping words into logical columns."""
        words = row.get('words', [])
        if not words:
            # Apply OCR correction to raw cells
            return [self._correct_ocr_assay_name(cell) for cell in row['cells']]

        # Group words that are close together horizontally into column headers
        # Words within 50 pixels are likely part of the same column header
        column_gap_threshold = 50
        columns = []
        current_col_words = []
        last_right = 0

        for word in words:
            word_left = word['left']
            word_right = word_left + word['width']

            if current_col_words and (word_left - last_right) > column_gap_threshold:
                # Start a new column
                col_text = ' '.join(w['text'] for w in current_col_words)
                columns.append(col_text)
                current_col_words = []

            current_col_words.append(word)
            last_right = word_right

        # Don't forget last column
        if current_col_words:
            col_text = ' '.join(w['text'] for w in current_col_words)
            columns.append(col_text)

        # Apply OCR correction to all column headers
        columns = [self._correct_ocr_assay_name(col) for col in columns]

        return columns

    def _find_data_rows_by_pattern(self, rows: List[dict], start_idx: int) -> List[Tuple[str, List[str]]]:
        """Find data rows by looking for compound ID + numeric values pattern.

        Returns list of (compound_id, [values]) tuples.
        """
        result = []

        # Look for rows that:
        # 1. Start with a number (compound ID)
        # 2. Have numeric values or symbols after it
        compound_pattern = re.compile(r'^(\d{1,4}(?:-\d{1,4})?[a-zA-Z]?)$')
        value_pattern = re.compile(r'^[<>±~]?\s*\d+\.?\d*\s*$|^[+＋]{1,5}$|^[-−–—]{1,5}$|^ND$|^NA$|^NT$', re.IGNORECASE)

        for row in rows[start_idx:]:
            cells = row['cells']
            if not cells:
                continue

            # Check if first cell looks like a compound ID
            first_cell = cells[0].strip() if cells else ""
            compound_match = compound_pattern.match(first_cell)

            if compound_match:
                compound_id = self._fix_ocr_compound_id(compound_match.group(1))

                # Count how many remaining cells look like values
                values = []
                for cell in cells[1:]:
                    cell_clean = cell.strip()
                    if value_pattern.match(cell_clean) or self._parse_value(cell_clean):
                        values.append(cell_clean)
                    else:
                        values.append('')  # Keep position for alignment

                if any(values):  # At least one value found
                    result.append((compound_id, values))
                    if len(result) <= 3:
                        self._debug_info.append(f"  Pattern match: Compound {compound_id} -> {values[:5]}...")

        return result

    def _parse_bio_text(self, text: str, page_num: int) -> Dict[str, BiologicalData]:
        """Parse OCR'd or extracted text for biological data."""
        data = {}

        lines = text.split('\n')

        # First pass: Look for header line with assay names
        header_info = self._find_header_line(lines)
        if header_info:
            self._debug_info.append(f"  Found header: {header_info}")

        # Pattern for compound number at start of line
        compound_line_pattern = re.compile(r'^\s*(\d{1,4}(?:-\d{1,4})?[a-zA-Z]?)\s+(.+)$')

        # Pattern for '+' symbol IC50 values (common OCR errors: H+, t+, a+, etc.)
        plus_pattern = re.compile(r'[HhtTaA4]?[+＋]{1,5}|[+＋]{1,5}')

        # Pattern for numeric IC50 values with units
        numeric_value_pattern = re.compile(
            r'[<>≤≥]?\s*\d+\.?\d*\s*(?:nM|μM|uM|mM|pM)',
            re.IGNORECASE
        )

        # Pattern for dash/inactive
        inactive_pattern = re.compile(r'^[-−–—]+$|^(?:ND|NA|NT|inactive)$', re.IGNORECASE)

        for line in lines:
            line = line.strip()
            if not line:
                continue

            match = compound_line_pattern.match(line)
            if match:
                compound_id = match.group(1)
                rest_of_line = match.group(2)

                # Filter out NMR notation (1H, 2H, 3H, 4H)
                if re.match(r'^\d[HhDd]$', compound_id):
                    continue

                # Clean up common OCR errors in + symbols
                cleaned = self._normalize_plus_symbols(rest_of_line)
                cleaned = re.sub(r'[Hh][+＋]', '+', cleaned)
                cleaned = re.sub(r'[Tt][+＋]', '+', cleaned)
                cleaned = re.sub(r'[Aa][+＋]', '+', cleaned)

                ic50_value = None

                # First try to find '+' symbol patterns
                plus_matches = plus_pattern.findall(cleaned)
                if plus_matches:
                    raw_plus = plus_matches[0]
                    plus_count = raw_plus.count('+') + raw_plus.count('＋')
                    if plus_count > 0:
                        ic50_value = '+' * min(plus_count, 5)

                # If no '+' pattern, try numeric with units
                if not ic50_value:
                    numeric_matches = numeric_value_pattern.findall(cleaned)
                    if numeric_matches:
                        ic50_value = numeric_matches[0].strip()

                # If still nothing, check for inactive/dash
                if not ic50_value:
                    inactive_matches = inactive_pattern.findall(cleaned)
                    if inactive_matches:
                        ic50_value = '-'

                if ic50_value:
                    self._debug_info.append(f"  OCR found compound {compound_id}: IC50={ic50_value}")

                    if compound_id not in data:
                        data[compound_id] = BiologicalData(compound_id=compound_id)

                    data[compound_id].ic50 = ic50_value

        self._debug_info.append(f"  OCR parsing found {len(data)} compounds")
        return data

    def _find_header_line(self, lines: List[str]) -> Optional[dict]:
        """Find and parse the header line containing assay column names."""
        assay_keywords = ['ic50', 'ic 50', 'ec50', 'ec 50', 'ki', 'kd',
                         'inhibition', 'activity', 'binding', 'potency']

        for idx, line in enumerate(lines):
            line_lower = line.lower()
            keyword_count = sum(1 for kw in assay_keywords if kw in line_lower)
            if keyword_count >= 1:
                columns = []
                parts = re.split(r'\s{2,}|\t', line)
                for part in parts:
                    part = part.strip()
                    if part:
                        columns.append(part)

                if columns:
                    return {
                        'line_index': idx,
                        'columns': columns,
                        'raw': line
                    }

        return None

    def _is_biological_table_data(self, table_data: List[List]) -> bool:
        """Check if table data contains biological data."""
        try:
            if not table_data or len(table_data) < 2:
                return False

            # Check header row and first few data rows for assay keywords
            text_to_check = []
            for row in table_data[:3]:
                text_to_check.extend(str(cell).lower() for cell in row if cell)

            combined_text = ' '.join(text_to_check)

            for keyword in self.TABLE_KEYWORDS:
                if keyword in combined_text:
                    self._debug_info.append(f"    Table matched keyword: '{keyword}'")
                    return True

            # Also check if any column header matches assay patterns
            header_row = table_data[0]
            for cell in header_row:
                if cell:
                    cell_str = str(cell)
                    for pattern_name, pattern in self.ASSAY_PATTERNS.items():
                        if pattern.search(cell_str):
                            self._debug_info.append(f"    Table matched pattern '{pattern_name}' in header: '{cell_str}'")
                            return True

            # Additional check: if table has numeric data with units, it might be bio data
            # Check for patterns like "10 nM", "5.2 μM", ">100", "<1", etc.
            for row in table_data[1:3]:  # Check first 2 data rows
                for cell in row:
                    if cell:
                        cell_str = str(cell)
                        # Check for numeric values with units or comparison operators
                        if re.search(r'[<>≤≥]?\s*\d+\.?\d*\s*[nμump]?M', cell_str, re.IGNORECASE):
                            self._debug_info.append(f"    Table has numeric value with units: '{cell_str}'")
                            return True
                        # Check for +/- symbols (common for activity)
                        if re.search(r'^[+＋]{1,5}$|^[-−–—]{1,5}$', cell_str.strip()):
                            self._debug_info.append(f"    Table has activity symbol: '{cell_str}'")
                            return True

            return False

        except Exception as e:
            self._debug_info.append(f"    Error in _is_biological_table_data: {e}")
            return False

    def _parse_biological_table_data(self, table_data: List[List]) -> Dict[str, BiologicalData]:
        """Parse biological data from table data.

        Extracts ALL columns into other_assays dict using actual header text as key.
        Also populates legacy fields (ic50, ec50, ki, kd) for backwards compatibility.
        """
        data = {}

        try:
            if not table_data or len(table_data) < 2:
                return data

            # Parse header to find column indices
            header_row = table_data[0]
            column_mapping = self._parse_header(header_row)

            self._debug_info.append(f"    Column mapping: {column_mapping}")

            # Get compound column index
            compound_idx = column_mapping.pop('_compound_idx', 0)
            self._debug_info.append(f"    Compound column index: {compound_idx}")
            self._debug_info.append(f"    Data columns: {list(column_mapping.keys())}")

            # Parse data rows
            for row_idx, row in enumerate(table_data[1:], start=1):
                compound_id = self._extract_compound_id(row, compound_idx)
                if not compound_id:
                    if row_idx <= 3:  # Debug first few rows
                        self._debug_info.append(f"    Row {row_idx}: No compound ID found in cell: '{row[compound_idx] if compound_idx < len(row) else 'N/A'}'")
                    continue

                bio_data = BiologicalData(compound_id=compound_id)

                # Process ALL columns - store in other_assays with actual header text
                for header_text, col_idx in column_mapping.items():
                    if col_idx < len(row):
                        raw_value = row[col_idx]
                        value = self._parse_value(raw_value)
                        if value:
                            # Store in other_assays with actual header text
                            bio_data.other_assays[header_text] = value

                            # Also populate legacy fields if pattern matches
                            if self.ASSAY_PATTERNS['ic50'].search(header_text):
                                bio_data.ic50 = value
                            elif self.ASSAY_PATTERNS['ec50'].search(header_text):
                                bio_data.ec50 = value
                            elif self.ASSAY_PATTERNS['ki'].search(header_text):
                                bio_data.ki = value
                            elif self.ASSAY_PATTERNS['kd'].search(header_text):
                                bio_data.kd = value

                if row_idx <= 3:  # Debug first few rows
                    self._debug_info.append(f"    Row {row_idx}: Compound '{compound_id}' -> {bio_data.other_assays}")

                data[compound_id] = bio_data

        except Exception as e:
            self._debug_info.append(f"    Parse error: {e}")
            import traceback
            self._debug_info.append(f"    Traceback: {traceback.format_exc()}")

        return data

    def _parse_header(self, header_row: List) -> Dict[str, int]:
        """Parse table header to map column names to indices.

        Captures ALL column names with their actual header text.
        Returns mapping of header_text -> column_index for all non-compound columns.
        """
        mapping = {}
        compound_idx = None

        # First pass: find compound column
        for idx, cell in enumerate(header_row):
            if cell is None:
                continue
            cell_text = str(cell).strip()
            if not cell_text:
                continue
            for pattern in self.COMPOUND_COL_PATTERNS:
                if pattern.search(cell_text):
                    compound_idx = idx
                    break
            if compound_idx is not None:
                break

        # If no compound column found, assume first column (index 0) is compound
        if compound_idx is None:
            compound_idx = 0

        # Second pass: collect all non-compound columns
        for idx, cell in enumerate(header_row):
            if idx == compound_idx:
                continue  # Skip compound column

            if cell is None:
                continue

            cell_text = str(cell).strip()
            if not cell_text:
                continue

            # Store ALL non-compound columns with their actual header text
            # Normalize whitespace and correct OCR errors
            clean_header = ' '.join(cell_text.split())
            clean_header = self._correct_ocr_assay_name(clean_header)
            if clean_header:
                mapping[clean_header] = idx

        # Store compound index
        mapping['_compound_idx'] = compound_idx

        return mapping

    def _extract_compound_id(self, row: List, col_idx: int) -> Optional[str]:
        """Extract compound ID from a table row."""
        if col_idx >= len(row):
            return None

        cell = row[col_idx]
        if cell is None:
            return None

        cell_text = str(cell).strip()
        if not cell_text:
            return None

        match = self.COMPOUND_ID_PATTERN.search(cell_text)
        if match:
            return match.group(1)

        cleaned = re.sub(r'[^\w]', '', cell_text)
        if len(cleaned) <= 10 and cleaned:
            return cleaned

        return None

    def _parse_value(self, cell) -> Optional[str]:
        """Parse an assay value from a table cell."""
        if cell is None:
            return None

        cell_text = str(cell).strip()

        # Clean up table border artifacts
        cell_text = cell_text.strip('|').strip()

        if not cell_text or cell_text == '-' or cell_text.lower() == 'none':
            return None

        # Skip obvious garbage (single punctuation, brackets, etc.)
        if cell_text in ('|', '||', '[', ']', '{', '}', '(', ')'):
            return None

        # Check for symbol patterns (+, ++, -, ND, etc.)
        match = self.SYMBOL_PATTERN.search(cell_text)
        if match:
            return match.group(1)

        # Check for numeric values with units
        match = self.VALUE_PATTERN.search(cell_text)
        if match:
            value = match.group(1).strip()
            unit = match.group(2) or ''
            if value:
                return f"{value} {unit}".strip()

        # Check for comparison operators (>1000, <10, etc.)
        if re.search(r'^[<>≤≥]\s*\d+', cell_text):
            return cell_text

        # Accept short alphanumeric strings (but not pure garbage)
        if len(cell_text) <= 20 and cell_text:
            # Must have at least one digit or be a known symbol
            if re.search(r'\d', cell_text) or cell_text.lower() in ('nd', 'na', 'nt', 'inactive', 'active'):
                return cell_text

        return None

    def _merge_bio_data(self, existing: BiologicalData, new: BiologicalData) -> None:
        """Merge new biological data into existing."""
        for attr in ['ic50', 'ec50', 'ki', 'kd']:
            new_value = getattr(new, attr)
            if new_value and not getattr(existing, attr):
                setattr(existing, attr, new_value)

        for key, value in new.other_assays.items():
            if key not in existing.other_assays:
                existing.other_assays[key] = value

    def get_debug_info(self) -> str:
        """Get debug information from the last extraction."""
        return "\n".join(self._debug_info)

    def extract_from_text_blocks(
        self,
        text_blocks: List[dict],
        page_num: int
    ) -> Dict[str, BiologicalData]:
        """Fallback: Extract biological data from text blocks."""
        data = {}

        full_text = ' '.join(block.get('text', '') for block in text_blocks)

        pattern = re.compile(
            r'(?:compound|cpd|ex\.?|example)\s*(\d+[a-z]?)\s*[:\-]?\s*'
            r'(ic50|ec50|ki|kd)\s*[=:\-]?\s*([<>≤≥±~]?\s*[\d.]+\s*(?:nM|μM|uM|mM|pM)?|\+{1,5})',
            re.IGNORECASE
        )

        for match in pattern.finditer(full_text):
            compound_id = self._fix_ocr_compound_id(match.group(1))
            assay_type = match.group(2).lower().replace(' ', '')
            value = match.group(3).strip()

            if compound_id not in data:
                data[compound_id] = BiologicalData(compound_id=compound_id)

            if 'ic50' in assay_type:
                data[compound_id].ic50 = value
            elif 'ec50' in assay_type:
                data[compound_id].ec50 = value
            elif 'ki' in assay_type:
                data[compound_id].ki = value
            elif 'kd' in assay_type:
                data[compound_id].kd = value

        return data
