"""Compound number detection for chemical structures in patent PDFs.

Uses OCR to extract text from regions adjacent to chemical structures.
"""

import re
from typing import List, Optional, Tuple
import numpy as np
from PIL import Image
import cv2

# Try to import pytesseract and configure Tesseract path
try:
    import pytesseract
    from ..utils.paths import configure_tesseract
    HAS_TESSERACT = configure_tesseract()
except ImportError:
    HAS_TESSERACT = False


class CompoundDetector:
    """Detects compound/example numbers near chemical structures in patent PDFs.

    Uses a combination of template matching and OCR to find compound labels
    in regions adjacent to detected chemical structures.
    """

    # Regex patterns for compound identifiers in OCR text
    COMPOUND_PATTERNS = [
        # Standalone numbers: "1", "12", "123"
        r'^\s*(\d{1,4})\s*$',
        # Numbers with letter suffix: "1a", "12b"
        r'^\s*(\d{1,4}[a-zA-Z])\s*$',
        # Parenthesized: "(1)", "(12)"
        r'^\s*\((\d{1,4}[a-zA-Z]?)\)\s*$',
        # Bracketed: "[1]", "[12]"
        r'^\s*\[(\d{1,4}[a-zA-Z]?)\]\s*$',
        # With prefix: "Cpd 1", "Ex 5", "Compound 1"
        r'(?:Cpd|Ex|Cmpd|Compound|Example|No)[\s\.\-:]*(\d{1,4}[a-zA-Z]?)',
        # Roman numerals
        r'^\s*([IVX]{1,4})\s*$',
    ]

    _compiled_patterns = None

    # Template matching parameters
    MATCH_THRESHOLD = 0.5  # Lower threshold for more matches
    SCALE_RANGE = (0.4, 1.6)
    SCALE_STEPS = 9

    # OCR region parameters (in pixels, relative to structure)
    REGION_WIDTH = 120  # Width of adjacent region to scan
    REGION_HEIGHT_FACTOR = 1.0  # Height as factor of structure height

    def __init__(self):
        """Initialize the compound detector."""
        if CompoundDetector._compiled_patterns is None:
            CompoundDetector._compiled_patterns = [
                re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                for pattern in self.COMPOUND_PATTERNS
            ]

    def _find_structure_position(
        self,
        structure_img: Image.Image,
        page_img: Image.Image
    ) -> Optional[Tuple[int, int, int, int]]:
        """Find structure position in page using template matching.

        Returns:
            Bounding box (x1, y1, x2, y2) or None if not found.
        """
        try:
            struct_array = np.array(structure_img.convert('L'))
            page_array = np.array(page_img.convert('L'))

            template_h, template_w = struct_array.shape
            page_h, page_w = page_array.shape

            if template_w >= page_w or template_h >= page_h:
                return None

            best_match = None
            best_score = self.MATCH_THRESHOLD

            scales = np.linspace(
                self.SCALE_RANGE[0],
                self.SCALE_RANGE[1],
                self.SCALE_STEPS
            )

            for scale in scales:
                new_w = int(template_w * scale)
                new_h = int(template_h * scale)

                if new_w >= page_w or new_h >= page_h or new_w < 20 or new_h < 20:
                    continue

                scaled_template = cv2.resize(struct_array, (new_w, new_h))

                try:
                    result = cv2.matchTemplate(
                        page_array,
                        scaled_template,
                        cv2.TM_CCOEFF_NORMED
                    )
                    _, max_val, _, max_loc = cv2.minMaxLoc(result)

                    if max_val > best_score:
                        best_score = max_val
                        x, y = max_loc
                        best_match = (x, y, x + new_w, y + new_h)
                except cv2.error:
                    continue

            return best_match

        except Exception:
            return None

    def _extract_region(
        self,
        page_img: Image.Image,
        struct_bbox: Tuple[int, int, int, int],
        position: str
    ) -> Optional[Image.Image]:
        """Extract a region adjacent to the structure.

        Args:
            page_img: Full page image.
            struct_bbox: Structure bounding box (x1, y1, x2, y2).
            position: 'left', 'right', 'above', or 'below'.

        Returns:
            Cropped region image or None if invalid.
        """
        x1, y1, x2, y2 = struct_bbox
        page_w, page_h = page_img.size
        struct_w = x2 - x1
        struct_h = y2 - y1

        region_w = self.REGION_WIDTH
        region_h = int(struct_h * self.REGION_HEIGHT_FACTOR)

        if position == 'left':
            rx1 = max(0, x1 - region_w)
            ry1 = y1
            rx2 = x1
            ry2 = y2
        elif position == 'right':
            rx1 = x2
            ry1 = y1
            rx2 = min(page_w, x2 + region_w)
            ry2 = y2
        elif position == 'above':
            rx1 = x1
            ry1 = max(0, y1 - region_h)
            rx2 = x2
            ry2 = y1
        elif position == 'below':
            rx1 = x1
            ry1 = y2
            rx2 = x2
            ry2 = min(page_h, y2 + region_h)
        else:
            return None

        # Ensure valid region
        if rx2 <= rx1 or ry2 <= ry1:
            return None

        # Minimum region size
        if (rx2 - rx1) < 20 or (ry2 - ry1) < 20:
            return None

        return page_img.crop((rx1, ry1, rx2, ry2))

    def _ocr_region(self, region_img: Image.Image) -> str:
        """Run OCR on a region image.

        Args:
            region_img: Image to OCR.

        Returns:
            Extracted text or empty string.
        """
        if not HAS_TESSERACT:
            return ""

        try:
            # Preprocess for better OCR
            img_array = np.array(region_img.convert('L'))

            # Apply threshold to make text clearer
            _, thresh = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed = Image.fromarray(thresh)

            # Run OCR with config for single line/word
            text = pytesseract.image_to_string(
                processed,
                config='--psm 7 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ()[]- '
            )
            return text.strip()
        except Exception:
            return ""

    def _parse_compound_id(self, text: str) -> Optional[str]:
        """Parse compound ID from OCR text.

        Args:
            text: OCR extracted text.

        Returns:
            Compound ID or None.
        """
        if not text:
            return None

        text = text.strip()

        # Reject obvious non-compound text
        # - Patent numbers (PCT, WO, US followed by numbers)
        # - Chemical formulas (multiple uppercase letters, subscripts)
        # - Too long strings
        if len(text) > 10:
            return None
        if re.search(r'(PCT|WO\d|US\d|[A-Z]{2,}[a-z]|[a-z][A-Z])', text):
            return None
        # Reject if it looks like a chemical formula (letters mixed with numbers in complex ways)
        if re.search(r'[A-Z][a-z]?\d+[A-Z]', text):  # Like "HoN2Se", "F5Cree"
            return None

        # Try each pattern for explicit compound labels
        for pattern in self._compiled_patterns:
            match = pattern.search(text)
            if match:
                # Extract just the number/ID part
                result = match.group(1) if match.lastindex else match.group(0)
                return result.strip()

        # Strict fallback: only accept pure numbers or number+single letter
        # Must be 1-4 digits optionally followed by single lowercase letter
        strict_match = re.match(r'^(\d{1,4})([a-z])?$', text)
        if strict_match:
            return text

        return None

    def _ocr_left_margin(
        self,
        page_img: Image.Image,
        num_structures: int
    ) -> List[Optional[str]]:
        """OCR the left margin of the page to find compound labels.

        This is a fallback when template matching fails.

        Args:
            page_img: Full page image.
            num_structures: Number of structures to find labels for.

        Returns:
            List of compound IDs (may contain None for unfound labels).
        """
        labels_with_pos = self._ocr_left_margin_with_positions(page_img)

        # Return first N labels (sorted by Y position)
        results = []
        for i in range(num_structures):
            if i < len(labels_with_pos):
                results.append(labels_with_pos[i][0])
            else:
                results.append(None)

        return results

    def _ocr_left_margin_with_positions(
        self,
        page_img: Image.Image
    ) -> List[Tuple[str, int]]:
        """OCR the left margin to find compound labels with their Y positions.

        Args:
            page_img: Full page image.

        Returns:
            List of (compound_id, y_position) tuples sorted by Y position.
        """
        if not HAS_TESSERACT:
            return []

        try:
            page_w, page_h = page_img.size

            # Extract left margin (first 15% of page width)
            margin_w = int(page_w * 0.15)
            margin_region = page_img.crop((0, 0, margin_w, page_h))

            # Preprocess
            img_array = np.array(margin_region.convert('L'))
            _, thresh = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed = Image.fromarray(thresh)

            # Run OCR with bounding box output
            data = pytesseract.image_to_data(
                processed,
                output_type=pytesseract.Output.DICT,
                config='--psm 6'
            )

            # Collect all potential labels with their y-position
            labels = []
            for i, text in enumerate(data['text']):
                text = text.strip()
                if not text:
                    continue

                compound_id = self._parse_compound_id(text)
                if compound_id:
                    y = data['top'][i]
                    h = data['height'][i]
                    # Use center Y position for more accurate matching
                    center_y = y + h // 2
                    labels.append((compound_id, center_y))

            # Sort by y-position (top to bottom)
            labels.sort(key=lambda x: x[1])

            return labels

        except Exception:
            return []

    def _detect_column_layout(
        self,
        structure_positions: List[Tuple[Optional[Tuple[int, int, int, int]], int]]
    ) -> List[Tuple[int, int]]:
        """Detect column layout from structure positions.

        Groups structures into columns by their X position.

        Args:
            structure_positions: List of (bbox, center_y) for each structure.

        Returns:
            List of (column_left_x, column_right_x) tuples defining each column.
        """
        # Collect left X positions of structures that have bounding boxes
        x_positions = []
        for bbox, _ in structure_positions:
            if bbox:
                x_positions.append(bbox[0])  # x1

        if not x_positions:
            return []

        # Cluster X positions to find columns
        # Sort and find gaps > 200px between positions
        x_sorted = sorted(set(x_positions))
        columns = []
        current_group = [x_sorted[0]]

        for i in range(1, len(x_sorted)):
            if x_sorted[i] - x_sorted[i - 1] > 200:
                # New column
                columns.append(current_group)
                current_group = [x_sorted[i]]
            else:
                current_group.append(x_sorted[i])

        columns.append(current_group)

        # For each column, compute the left edge (min x1 of structures in that column)
        column_ranges = []
        for group in columns:
            col_left = min(group)
            column_ranges.append(col_left)

        return column_ranges

    def _ocr_compound_columns(
        self,
        page_img: Image.Image,
        column_x_positions: List[int]
    ) -> List[Tuple[str, int, int]]:
        """OCR the compound number columns for each structure column.

        For each structure column, OCR a narrow strip to its left where
        compound numbers appear in patent tables.

        Args:
            page_img: Full page image.
            column_x_positions: Left X position of each structure column.

        Returns:
            List of (compound_id, y_position, column_x) tuples.
        """
        if not HAS_TESSERACT:
            return []

        all_labels = []
        page_w, page_h = page_img.size

        # Compound number column width (pixels to the left of structure column)
        CPD_COL_WIDTH = 160

        for col_x in column_x_positions:
            # The compound number strip is to the left of the structure column
            strip_x2 = col_x
            strip_x1 = max(0, col_x - CPD_COL_WIDTH)

            # Skip if strip is too narrow
            if strip_x2 - strip_x1 < 30:
                continue

            # Crop the compound number strip (full page height)
            strip_img = page_img.crop((strip_x1, 0, strip_x2, page_h))

            try:
                # Preprocess
                img_array = np.array(strip_img.convert('L'))
                _, thresh = cv2.threshold(
                    img_array, 0, 255,
                    cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                processed = Image.fromarray(thresh)

                # Run OCR with bounding box data
                data = pytesseract.image_to_data(
                    processed,
                    output_type=pytesseract.Output.DICT,
                    config='--psm 6'
                )

                for i, text in enumerate(data['text']):
                    text = text.strip()
                    if not text:
                        continue

                    compound_id = self._parse_compound_id(text)
                    if compound_id:
                        y = data['top'][i]
                        h = data['height'][i]
                        center_y = y + h // 2
                        all_labels.append((compound_id, center_y, col_x))

            except Exception:
                continue

        return all_labels

    def detect_compound_ids_batch(
        self,
        structure_images: List[Image.Image],
        page_img: Image.Image,
        text_blocks: List[dict],
        image_dpi: int = 200,
        margin: int = 150
    ) -> List[Tuple[Optional[str], Optional[Tuple[int, int, int, int]]]]:
        """Detect compound IDs for multiple structures using OCR.

        Strategy:
        1. Template matching to find each structure's position
        2. Detect column layout from structure positions
        3. OCR the compound number strip for each column
        4. Match structures to compound IDs by Y-position + column proximity

        Args:
            structure_images: List of cropped structure images.
            page_img: PIL Image of the full page.
            text_blocks: Text blocks (used as fallback, may be empty for image-based PDFs).
            image_dpi: DPI of the rendered page image.
            margin: Not used in current implementation.

        Returns:
            List of (compound_id, bounding_box) tuples, one per structure.
        """
        num_structures = len(structure_images)

        if not structure_images or page_img is None:
            return [(None, None)] * num_structures

        if not HAS_TESSERACT:
            return self._text_based_fallback(structure_images, text_blocks)

        # Step 1: Find structure positions via template matching
        page_h = page_img.size[1]
        structure_positions = []

        for idx, struct_img in enumerate(structure_images):
            bbox = self._find_structure_position(struct_img, page_img)
            if bbox:
                x1, y1, x2, y2 = bbox
                center_y = (y1 + y2) // 2
                structure_positions.append((bbox, center_y))
            else:
                estimated_y = int((idx + 0.5) * page_h / num_structures)
                structure_positions.append((None, estimated_y))

        # Step 2: Detect column layout
        column_x_positions = self._detect_column_layout(structure_positions)

        # Step 3: OCR compound number strips for each column
        if column_x_positions:
            column_labels = self._ocr_compound_columns(page_img, column_x_positions)
        else:
            column_labels = []

        # Step 4: Match structures to compound IDs
        results = []
        used_label_indices = set()

        # Assign each structure's column based on its X position
        for struct_idx, (bbox, struct_y) in enumerate(structure_positions):
            compound_id = None
            best_match_idx = None
            best_distance = float('inf')

            # Determine which column this structure belongs to
            struct_col_x = None
            if bbox:
                struct_x = bbox[0]
                # Find the closest column
                for col_x in column_x_positions:
                    if abs(struct_x - col_x) < 100:
                        struct_col_x = col_x
                        break

            # Find the closest compound ID in the same column
            for label_idx, (label_id, label_y, label_col_x) in enumerate(column_labels):
                if label_idx in used_label_indices:
                    continue

                # Must be in the same column (or no column constraint)
                if struct_col_x is not None and label_col_x != struct_col_x:
                    continue

                distance = abs(struct_y - label_y)
                if distance < best_distance:
                    best_distance = distance
                    best_match_idx = label_idx
                    compound_id = label_id

            # Only match if reasonably close (within half the average row height)
            max_distance = max(200, page_h // (num_structures + 1))
            if best_match_idx is not None and best_distance < max_distance:
                used_label_indices.add(best_match_idx)
            else:
                compound_id = None

            results.append((compound_id, bbox))

        # Fallback: if few matches, try the old left margin approach
        matched_count = sum(1 for r in results if r[0] is not None)
        if matched_count < num_structures // 2:
            margin_labels = self._ocr_left_margin_with_positions(page_img)
            if margin_labels:
                # Try to fill in missing IDs
                used_margin = set()
                for idx, (cid, bbox) in enumerate(results):
                    if cid is not None:
                        continue
                    _, struct_y = structure_positions[idx]
                    best_m_idx = None
                    best_m_dist = float('inf')
                    for m_idx, (m_id, m_y) in enumerate(margin_labels):
                        if m_idx in used_margin:
                            continue
                        dist = abs(struct_y - m_y)
                        if dist < best_m_dist:
                            best_m_dist = dist
                            best_m_idx = m_idx
                    if best_m_idx is not None and best_m_dist < max_distance:
                        used_margin.add(best_m_idx)
                        results[idx] = (margin_labels[best_m_idx][0], bbox)

        return results

    def assign_sequential_ids(
        self,
        structure_bboxes: List[Optional[Tuple[int, int, int, int]]],
        start_id: int = 1,
        row_tolerance: int = 100
    ) -> List[str]:
        """Assign sequential compound IDs based on reading order.

        For two-column patent layouts, reading order is:
        - Group structures into rows by Y position
        - Within each row, order left to right
        - Assign sequential IDs: 1, 2, 3, ...

        Args:
            structure_bboxes: List of bounding boxes (x1, y1, x2, y2) for each structure.
            start_id: Starting compound ID number.
            row_tolerance: Y-distance threshold for grouping into same row.

        Returns:
            List of compound ID strings in the same order as input.
        """
        if not structure_bboxes:
            return []

        num_structures = len(structure_bboxes)

        # Build list of (original_index, center_x, center_y)
        indexed_positions = []
        for idx, bbox in enumerate(structure_bboxes):
            if bbox:
                x1, y1, x2, y2 = bbox
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                indexed_positions.append((idx, center_x, center_y))
            else:
                # No bbox - use index-based estimate
                indexed_positions.append((idx, idx * 100, idx * 100))

        # Sort by Y first to group into rows
        indexed_positions.sort(key=lambda x: x[2])

        # Group into rows (structures within row_tolerance Y are in same row)
        rows = []
        current_row = [indexed_positions[0]]
        for i in range(1, len(indexed_positions)):
            _, _, prev_y = current_row[-1]
            _, _, curr_y = indexed_positions[i]
            if abs(curr_y - prev_y) <= row_tolerance:
                current_row.append(indexed_positions[i])
            else:
                rows.append(current_row)
                current_row = [indexed_positions[i]]
        rows.append(current_row)

        # Within each row, sort by X (left to right)
        for row in rows:
            row.sort(key=lambda x: x[1])

        # Flatten and assign IDs in reading order
        reading_order = []
        for row in rows:
            reading_order.extend(row)

        # Create result array mapping original index to compound ID
        results = [None] * num_structures
        for position, (orig_idx, _, _) in enumerate(reading_order):
            compound_id = str(start_id + position)
            results[orig_idx] = compound_id

        return results

    def _text_based_fallback(
        self,
        structure_images: List[Image.Image],
        text_blocks: List[dict]
    ) -> List[Tuple[Optional[str], Optional[Tuple[int, int, int, int]]]]:
        """Fallback to text-based detection when Tesseract is not available."""
        num_structures = len(structure_images)

        if not text_blocks:
            return [(None, None)] * num_structures

        # Extract potential labels from text blocks
        labels = []
        for block in text_blocks:
            text = block.get('text', '').strip()
            if text and len(text) <= 20:
                compound_id = self._parse_compound_id(text)
                if compound_id:
                    bbox = block.get('bbox', (0, 0, 0, 0))
                    labels.append((compound_id, bbox[1]))  # Store with y-position

        # Sort by y-position
        labels.sort(key=lambda x: x[1])

        # Assign to structures
        results = []
        for i in range(num_structures):
            if i < len(labels):
                results.append((labels[i][0], None))
            else:
                results.append((None, None))

        return results

    def detect_compound_id(
        self,
        structure_img: Image.Image,
        page_img: Image.Image,
        text_blocks: List[dict],
        image_dpi: int = 200,
        margin: int = 150
    ) -> Tuple[Optional[str], Optional[Tuple[int, int, int, int]]]:
        """Detect compound ID for a single structure."""
        results = self.detect_compound_ids_batch(
            [structure_img],
            page_img,
            text_blocks,
            image_dpi=image_dpi,
            margin=margin
        )
        return results[0] if results else (None, None)

    def debug_text_blocks(self, text_blocks: List[dict]) -> str:
        """Generate debug output for text blocks."""
        lines = ["=== Text Blocks Debug ==="]
        lines.append(f"Tesseract available: {HAS_TESSERACT}")

        for i, block in enumerate(text_blocks):
            text = block.get('text', '').strip()
            bbox = block.get('bbox', (0, 0, 0, 0))
            compound_id = self._parse_compound_id(text) if text else None
            status = "LABEL" if compound_id else "text"
            lines.append(f"[{i}] ({status}) bbox={bbox[:2]} text='{text[:50]}'")

        return "\n".join(lines)

    def debug_compound_detection(
        self,
        structure_images: List[Image.Image],
        page_img: Image.Image
    ) -> str:
        """Generate detailed debug output for compound detection.

        Args:
            structure_images: List of structure images.
            page_img: Full page image.

        Returns:
            Debug information string.
        """
        lines = ["=== Compound Detection Debug ==="]
        lines.append(f"Tesseract available: {HAS_TESSERACT}")
        lines.append(f"Number of structures: {len(structure_images)}")
        lines.append(f"Page size: {page_img.size if page_img else 'None'}")
        lines.append("")

        if not HAS_TESSERACT:
            lines.append("Tesseract not available - skipping OCR debug")
            return "\n".join(lines)

        # Debug structure positions
        lines.append("--- Structure Positions ---")
        page_h = page_img.size[1] if page_img else 0
        num_structures = len(structure_images)
        structure_positions = []

        for idx, struct_img in enumerate(structure_images):
            bbox = self._find_structure_position(struct_img, page_img)
            if bbox:
                x1, y1, x2, y2 = bbox
                center_y = (y1 + y2) // 2
                structure_positions.append((bbox, center_y))
                lines.append(f"  Structure {idx}: bbox={bbox}, center_y={center_y}")
            else:
                estimated_y = int((idx + 0.5) * page_h / num_structures) if num_structures > 0 else 0
                structure_positions.append((None, estimated_y))
                lines.append(f"  Structure {idx}: NOT FOUND (estimated_y={estimated_y})")
        lines.append("")

        # Debug column layout detection
        lines.append("--- Column Layout ---")
        column_x_positions = self._detect_column_layout(structure_positions)
        lines.append(f"Detected {len(column_x_positions)} columns at X positions: {column_x_positions}")
        lines.append("")

        # Debug compound column OCR
        lines.append("--- Compound Column OCR ---")
        if column_x_positions:
            column_labels = self._ocr_compound_columns(page_img, column_x_positions)
            lines.append(f"Found {len(column_labels)} compound labels:")
            for i, (label, y, col_x) in enumerate(column_labels):
                lines.append(f"  [{i}] ID='{label}' at Y={y}, column_x={col_x}")
        else:
            lines.append("No columns detected")
        lines.append("")

        # Debug matching
        lines.append("--- Final Matching ---")
        results = self.detect_compound_ids_batch(
            structure_images, page_img, [], image_dpi=200
        )
        for idx, (compound_id, bbox) in enumerate(results):
            lines.append(f"  Structure {idx} -> Compound ID: {compound_id or 'None'}")

        return "\n".join(lines)
