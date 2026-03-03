"""Background worker for processing PDFs without blocking the UI."""

import logging
import os
import re
import time
from typing import List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

from PySide6.QtCore import QThread, Signal, QMutex, QMutexLocker

from ..models.extraction_result import ExtractionResult, ProcessingProgress
from ..core.pdf_processor import PDFProcessor
from ..core.structure_detector import StructureDetector
from ..core.smiles_predictor import SMILESPredictor
from ..core.smiles_validator import SMILESValidator
from ..core.compound_detector import CompoundDetector
from ..core.biological_data_extractor import BiologicalDataExtractor
from ..core.inference_settings import InferenceSettings, InferenceMode
from ..core.inference_provider import InferenceProvider

# Try to import pytesseract for OCR-based compound number detection
try:
    import pytesseract
    from PIL import Image
    from ..utils.paths import configure_tesseract
    HAS_TESSERACT = configure_tesseract()
except ImportError:
    HAS_TESSERACT = False


def parse_page_range(text: str, max_pages: int) -> Set[int]:
    """Parse a page range string into a set of 1-indexed page numbers.

    Accepts formats like: "1-10", "5,8,12", "1-5, 20-25", or blank for all pages.

    Args:
        text: Page range string (e.g., "1-5, 8, 12-15").
        max_pages: Maximum valid page number.

    Returns:
        Set of 1-indexed page numbers. Empty set if text is blank (meaning all pages).

    Raises:
        ValueError: If the format is invalid or page numbers are out of range.
    """
    text = text.strip()
    if not text:
        return set()

    pages = set()
    for part in text.split(','):
        part = part.strip()
        if not part:
            continue

        if '-' in part:
            bounds = part.split('-', 1)
            try:
                start = int(bounds[0].strip())
                end = int(bounds[1].strip())
            except ValueError:
                raise ValueError(f"Invalid range: '{part}'")
            if start < 1 or end < 1:
                raise ValueError(f"Page numbers must be >= 1: '{part}'")
            if start > max_pages or end > max_pages:
                raise ValueError(f"Page number exceeds total pages ({max_pages}): '{part}'")
            if start > end:
                raise ValueError(f"Invalid range (start > end): '{part}'")
            pages.update(range(start, end + 1))
        else:
            try:
                page = int(part)
            except ValueError:
                raise ValueError(f"Invalid page number: '{part}'")
            if page < 1 or page > max_pages:
                raise ValueError(f"Page {page} out of range (1-{max_pages})")
            pages.add(page)

    return pages


class ProcessingWorker(QThread):
    """Background worker thread for PDF processing.

    Signals:
        progress_updated: Emitted when processing progress changes.
        result_ready: Emitted when a single structure result is ready.
        processing_complete: Emitted when all processing is done.
        error_occurred: Emitted when an error occurs.
    """

    # Signals
    progress_updated = Signal(ProcessingProgress)
    result_ready = Signal(ExtractionResult)
    processing_complete = Signal(list)  # List[ExtractionResult]
    error_occurred = Signal(str)
    warning_occurred = Signal(str)

    def __init__(self, parent=None):
        """Initialize the processing worker."""
        super().__init__(parent)
        self._pdf_paths: List[str] = []
        self._page_filter: Optional[Set[int]] = None
        self._cancel_requested = False
        self._high_accuracy_mode: bool = False
        self._mutex = QMutex()

        # Lazy-initialized processors
        self._pdf_processor: Optional[PDFProcessor] = None
        self._structure_detector: Optional[StructureDetector] = None
        self._smiles_predictor: Optional[SMILESPredictor] = None
        self._smiles_validator: Optional[SMILESValidator] = None
        self._compound_detector: Optional[CompoundDetector] = None
        self._bio_data_extractor: Optional[BiologicalDataExtractor] = None

        # Cloud/local inference provider
        self._inference_provider: Optional[InferenceProvider] = None
        self._inference_settings = InferenceSettings.get_instance()

        # Auto page detection
        self._auto_detect_pages: bool = True

        # Store extracted bio data for access after processing
        # Format: {source_file: {compound_id: BiologicalData}}
        self._extracted_bio_data: dict = {}

    def set_pdf_path(self, path: str) -> None:
        """Set a single PDF file path to process (legacy support).

        Args:
            path: Path to the PDF file.
        """
        self._pdf_paths = [path]

    def set_pdf_paths(self, paths: List[str]) -> None:
        """Set multiple PDF file paths to process.

        Args:
            paths: List of paths to PDF files.
        """
        self._pdf_paths = paths

    def set_high_accuracy_mode(self, enabled: bool) -> None:
        """Enable or disable high accuracy mode.

        When enabled, uses higher DPI rendering and image preprocessing
        for better accuracy on complex structures like macrocycles.

        Args:
            enabled: True to enable high accuracy mode.
        """
        self._high_accuracy_mode = enabled

    def set_page_filter(self, page_filter: Optional[Set[int]]) -> None:
        """Set page filter to process only specific pages.

        Args:
            page_filter: Set of 1-indexed page numbers to process,
                         or None/empty set for all pages.
        """
        self._page_filter = page_filter if page_filter else None

    def set_auto_detect_pages(self, enabled: bool) -> None:
        """Enable or disable automatic page detection before processing.

        When enabled, uses PageClassifier to scan pages and skip
        text-only pages that don't contain chemical structures.

        Args:
            enabled: True to enable auto page detection.
        """
        self._auto_detect_pages = enabled

    def request_cancel(self) -> None:
        """Request cancellation of the current processing."""
        with QMutexLocker(self._mutex):
            self._cancel_requested = True

    def _is_cancelled(self) -> bool:
        """Check if cancellation was requested (thread-safe)."""
        with QMutexLocker(self._mutex):
            return self._cancel_requested

    def _init_processors(self) -> None:
        """Initialize processing components."""
        if self._pdf_processor is None:
            # Use higher DPI for high accuracy mode (350 vs 200)
            dpi = 350 if self._high_accuracy_mode else 200
            self._pdf_processor = PDFProcessor(dpi=dpi)

        # Initialize inference provider (handles local/cloud switching)
        if self._inference_provider is None:
            self._inference_provider = InferenceProvider()

        # For local DECIMER inference, also initialize direct references (used by some code paths)
        # Skip for cloud and lightweight modes — they don't need TensorFlow/DECIMER
        if not self._inference_settings.is_cloud and not self._inference_settings.is_lightweight:
            if self._structure_detector is None:
                self._structure_detector = StructureDetector()
            if self._smiles_predictor is None:
                self._smiles_predictor = SMILESPredictor()

        if self._smiles_validator is None:
            self._smiles_validator = SMILESValidator()
        if self._compound_detector is None:
            self._compound_detector = CompoundDetector()
        if self._bio_data_extractor is None:
            self._bio_data_extractor = BiologicalDataExtractor()

    def run(self) -> None:
        """Main processing loop - runs in background thread."""
        # Reset cancel flag
        with QMutexLocker(self._mutex):
            self._cancel_requested = False

        if not self._pdf_paths:
            self.error_occurred.emit("No PDF files specified")
            return

        results: List[ExtractionResult] = []
        total_files = len(self._pdf_paths)
        overall_total_pages = 0
        overall_current_page = 0

        try:
            # Initialize processors
            self._emit_progress(0, 1, 0, 0, "Initializing processors...")
            self._init_processors()

            # Time tracking for ETA estimation
            self._processing_start = time.time()
            self._pages_processed = 0

            # Process each PDF file
            for file_idx, pdf_path in enumerate(self._pdf_paths):
                if self._is_cancelled():
                    break

                file_name = os.path.basename(pdf_path)
                file_num = file_idx + 1

                # Open PDF
                self._emit_progress(
                    0, 1, 0, 0,
                    f"File {file_num}/{total_files}: Opening {file_name}..."
                )
                pdf_info = self._pdf_processor.open(pdf_path)
                total_pages = pdf_info.page_count
                overall_total_pages += total_pages

                # Determine which pages to process
                # If user provided a manual page range, respect it (skip auto-detect)
                # If no manual range, use auto-detect to skip text-only pages
                effective_page_filter = self._page_filter
                section_bounds = None

                # Step 1: Patent section detection (fast, narrows to Examples section)
                if not self._page_filter:
                    try:
                        from ..core.patent_section_detector import PatentSectionDetector

                        def section_progress(msg):
                            if not self._is_cancelled():
                                self._emit_progress(
                                    0, total_pages, 0, 0,
                                    f"File {file_num}/{total_files}: {msg}"
                                )

                        detector = PatentSectionDetector()
                        section_bounds = detector.detect(
                            pdf_path, progress_callback=section_progress
                        )
                        if section_bounds.is_valid:
                            effective_page_filter = section_bounds.get_page_range()
                            self._emit_progress(
                                0, total_pages, 0, 0,
                                f"File {file_num}/{total_files}: {section_bounds.summary()}"
                            )
                    except Exception as e:
                        logger.debug("Patent section detection failed: %s", e)

                # Step 2: Auto-detect structure pages (visual scan)
                if self._auto_detect_pages and not self._page_filter:
                    self._emit_progress(
                        0, total_pages, 0, 0,
                        f"File {file_num}/{total_files}: Scanning pages for structures..."
                    )
                    try:
                        from ..core.page_classifier import get_classifier
                        classifier = get_classifier()

                        def scan_progress(current, total):
                            if self._is_cancelled():
                                return
                            self._emit_progress(
                                current, total, 0, 0,
                                f"File {file_num}/{total_files}: Scanning page {current}/{total}..."
                            )

                        auto_detected = set(classifier.detect_structure_pages(
                            pdf_path, progress_callback=scan_progress
                        ))

                        if auto_detected:
                            if effective_page_filter:
                                # Intersect with section bounds
                                effective_page_filter = effective_page_filter & auto_detected
                            else:
                                effective_page_filter = auto_detected
                            self._emit_progress(
                                0, total_pages, 0, 0,
                                f"File {file_num}/{total_files}: Found {len(effective_page_filter)} "
                                f"structure pages out of {total_pages}"
                            )
                        else:
                            msg = f"{file_name}: No structure pages detected. Skipping file."
                            self.warning_occurred.emit(msg)
                            self._emit_progress(
                                total_pages, total_pages, 0, 0, msg
                            )
                            self._pdf_processor.close()
                            continue

                    except Exception as e:
                        # Auto-detection failed — fall back to section bounds or all pages
                        self._emit_progress(
                            0, total_pages, 0, 0,
                            f"File {file_num}/{total_files}: Page scan failed ({e}), processing all pages..."
                        )

                # Track compound ID for sequential assignment across pages
                next_compound_id = 1

                # Show which pages will be processed
                if effective_page_filter:
                    page_count = len(effective_page_filter)
                    self._emit_progress(
                        0, total_pages, 0, 0,
                        f"File {file_num}/{total_files}: Processing {page_count} pages..."
                    )
                else:
                    self._emit_progress(
                        0, total_pages, 0, 0,
                        f"File {file_num}/{total_files}: Processing all {total_pages} pages..."
                    )

                # Process each page
                for page_num in range(1, total_pages + 1):
                    if self._is_cancelled():
                        self._emit_progress(page_num, total_pages, 0, 0, "Cancelled")
                        break

                    # Skip pages not in filter
                    if effective_page_filter and page_num not in effective_page_filter:
                        continue

                    overall_current_page += 1
                    # Build status message with ETA if available
                    status_msg = f"File {file_num}/{total_files} ({file_name}): Page {page_num}/{total_pages}"
                    if self._pages_processed > 0:
                        elapsed = time.time() - self._processing_start
                        avg_per_page = elapsed / self._pages_processed
                        remaining_pages = total_pages - page_num
                        remaining_seconds = avg_per_page * remaining_pages
                        status_msg += f" \u2014 {self._format_eta(remaining_seconds)}"
                    self._emit_progress(
                        page_num, total_pages, 0, 0, status_msg
                    )

                    # Render page as image
                    page_image = self._pdf_processor.get_page_image(page_num)

                    if self._is_cancelled():
                        break

                    # Detect chemical structures (uses cloud or local based on settings)
                    mode_label = "cloud" if self._inference_settings.is_cloud else "local"
                    self._emit_progress(
                        page_num, total_pages, 0, 0,
                        f"File {file_num}/{total_files}: Detecting structures on page {page_num} ({mode_label})..."
                    )
                    structures_with_boxes = self._inference_provider.detect_structures_with_boxes(page_image)

                    if not structures_with_boxes:
                        self._pages_processed += 1
                        continue

                    structures = [img for img, _ in structures_with_boxes]
                    structure_boxes = [box for _, box in structures_with_boxes]
                    total_structures = len(structures)

                    # Classify structures using Claude Haiku (if API key available)
                    compound_types = self._classify_structures(
                        page_image, structure_boxes
                    )

                    # Detect example numbers from page using OCR
                    # Falls back to sequential numbering if OCR is unavailable
                    compound_results = self._detect_example_numbers_from_page(
                        page_image, structures, next_compound_id
                    )

                    # Update next_compound_id: use max detected ID + 1 to avoid collisions
                    detected_ids = []
                    for cid, _ in compound_results:
                        try:
                            detected_ids.append(int(re.sub(r'[a-zA-Z]', '', cid)))
                        except ValueError:
                            pass
                    if detected_ids:
                        next_compound_id = max(max(detected_ids) + 1, next_compound_id + total_structures)
                    else:
                        next_compound_id += total_structures

                    # Process each detected structure
                    for struct_idx, struct_image in enumerate(structures):
                        if self._is_cancelled():
                            break

                        self._emit_progress(
                            page_num, total_pages, struct_idx + 1, total_structures,
                            f"File {file_num}/{total_files}: Page {page_num}, structure {struct_idx + 1}/{total_structures}"
                        )

                        # Create result object with source file
                        result = ExtractionResult(
                            page_number=page_num,
                            structure_index=struct_idx,
                            source_file=file_name,
                            original_image=struct_image
                        )

                        # Get compound ID and bounding box from batch results
                        if struct_idx < len(compound_results):
                            compound_id, bbox = compound_results[struct_idx]
                            result.compound_id = compound_id
                            result.bounding_box = bbox

                        # Set compound type from LLM classifier
                        if compound_types and struct_idx < len(compound_types):
                            result.compound_type = compound_types[struct_idx]

                        try:
                            # Predict SMILES (uses cloud or local based on settings)
                            smiles = self._inference_provider.predict_smiles(
                                struct_image, high_accuracy=self._high_accuracy_mode
                            )
                            result.smiles = smiles

                            if smiles:
                                # Validate and render
                                is_valid, canonical, rdkit_img = \
                                    self._smiles_validator.validate_and_render(smiles)
                                result.is_valid = is_valid
                                result.canonical_smiles = canonical
                                result.rdkit_image = rdkit_img

                                # Calculate physicochemical properties for valid SMILES
                                if is_valid:
                                    props = self._smiles_validator.get_all_properties(
                                        canonical or smiles
                                    )
                                    result.molecular_weight = props['molecular_weight']
                                    result.molecular_formula = props['molecular_formula']
                                    result.clogp = props['clogp']
                                    result.tpsa = props['tpsa']
                                    result.num_rotatable_bonds = props['num_rotatable_bonds']
                                    result.num_stereocenters = props['num_stereocenters']
                            else:
                                result.error_message = "SMILES prediction failed"

                        except Exception as e:
                            result.error_message = str(e)

                        results.append(result)
                        self.result_ready.emit(result)

                    self._pages_processed += 1

                # Extract biological data from the entire PDF
                if not self._is_cancelled():
                    self._emit_progress(
                        total_pages, total_pages, 0, 0,
                        f"File {file_num}/{total_files}: Extracting biological data..."
                    )
                    try:
                        # Use user's original page filter if set; otherwise use
                        # section bounds to limit bio data extraction to Examples section
                        bio_page_filter = self._page_filter
                        if not bio_page_filter and section_bounds and section_bounds.is_valid:
                            bio_page_filter = section_bounds.get_page_range()
                        bio_data = self._bio_data_extractor.extract_from_pdf(
                            pdf_path, page_filter=bio_page_filter
                        )

                        if bio_data:
                            # Store bio data for later access, keyed by source file
                            self._extracted_bio_data[file_name] = bio_data

                            # Log before merge
                            self._emit_progress(
                                total_pages, total_pages, 0, 0,
                                f"File {file_num}/{total_files}: Found {len(bio_data)} compounds, merging..."
                            )

                            # Merge biological data into results by compound ID
                            merge_count = self._merge_biological_data(results, bio_data, file_name)

                            # Verify merge worked
                            results_with_bio = sum(1 for r in results if r.bio_data)
                            self._emit_progress(
                                total_pages, total_pages, 0, 0,
                                f"File {file_num}/{total_files}: Merged {merge_count} of {len(bio_data)}, {results_with_bio} results have bio_data"
                            )
                        else:
                            self._emit_progress(
                                total_pages, total_pages, 0, 0,
                                f"File {file_num}/{total_files}: No bio data found"
                            )
                    except Exception as e:
                        import traceback
                        # Log the error but continue - biological data extraction is optional
                        self._emit_progress(
                            total_pages, total_pages, 0, 0,
                            f"File {file_num}/{total_files}: Bio data extraction error: {e}"
                        )

                # Close current PDF before opening next
                self._pdf_processor.close()

        except Exception as e:
            self.error_occurred.emit(f"Processing error: {e}")

        finally:
            # Clean up
            if self._pdf_processor:
                self._pdf_processor.close()

        # Emit completion
        if not self._is_cancelled():
            self._emit_progress(
                overall_current_page, overall_total_pages,
                0, 0, "Processing complete"
            )

            # Final debug: Check bio_data in results before emitting
            results_with_bio = [r for r in results if r.bio_data]
            if self._bio_data_extractor:
                self._bio_data_extractor._debug_info.append(f"\n=== FINAL CHECK ===")
                self._bio_data_extractor._debug_info.append(f"Results with bio_data: {len(results_with_bio)} / {len(results)}")
                if results_with_bio:
                    first_bio = results_with_bio[0]
                    self._bio_data_extractor._debug_info.append(f"First result bio_data: {first_bio.bio_data}")
                else:
                    self._bio_data_extractor._debug_info.append("No results have bio_data populated!")
                    # Debug: show first result's info
                    if results:
                        r = results[0]
                        self._bio_data_extractor._debug_info.append(f"First result: id={r.compound_id}, source={r.source_file}")

        self.processing_complete.emit(results)

    def _detect_example_numbers_from_page(
        self,
        page_image,
        structures: list,
        fallback_start_id: int
    ) -> List[Tuple[str, Optional[tuple]]]:
        """Detect example/compound numbers from the page using OCR.

        For patent tables with layout like "Example | Structure | IC50",
        runs full-page OCR at native resolution and filters for numbers
        in the left margin (x < 20% of page width), then matches each
        structure to the nearest number by Y-position.

        Args:
            page_image: PIL Image of the full PDF page (already at 200dpi).
            structures: List of detected structure images.
            fallback_start_id: Starting ID for sequential fallback.

        Returns:
            List of (compound_id, bounding_box) tuples, one per structure.
        """
        total_structures = len(structures)

        if not HAS_TESSERACT or not structures:
            return [(str(fallback_start_id + i), None) for i in range(total_structures)]

        try:
            width, height = page_image.size
            # x threshold: example numbers can be at 13-25% from left
            x_threshold = int(width * 0.28)

            # OCR the full page at native resolution (already 200dpi)
            ocr_data = pytesseract.image_to_data(
                page_image, output_type=pytesseract.Output.DICT
            )

            # Extract example numbers: digits in the left margin
            # Handles: "7", "100", "100-7", "100-11", "12a"
            example_number_pattern = re.compile(
                r'^(\d{1,4}(?:-\d{1,4})?[a-zA-Z]?)$'
            )
            detected_numbers = []  # [(number_str, y_center)]

            # Skip header area (top 12% has page numbers, patent IDs)
            y_header_threshold = int(height * 0.12)

            n_boxes = len(ocr_data['text'])
            for i in range(n_boxes):
                text = str(ocr_data['text'][i]).strip()
                conf = int(ocr_data['conf'][i]) if ocr_data['conf'][i] != '-1' else 0

                if not text or conf < 30:
                    continue

                x = ocr_data['left'][i]
                y = ocr_data['top'][i]
                if x > x_threshold or y < y_header_threshold:
                    continue

                # First try exact match
                match = example_number_pattern.match(text)

                # If no match, try OCR corrections for short texts only
                if not match and len(text) <= 6:
                    corrected = text
                    corrected = re.sub(r'(?<=\d)[ilI]', '1', corrected)
                    corrected = re.sub(r'[ilI](?=\d)', '1', corrected)
                    if re.match(r'^[ilI]{1,4}$', text):
                        corrected = re.sub(r'[ilI]', '1', text)
                    corrected = re.sub(r'[O]', '0', corrected)
                    match = example_number_pattern.match(corrected)
                if match:
                    number_str = match.group(1)
                    y_center = ocr_data['top'][i] + ocr_data['height'][i] // 2
                    detected_numbers.append((number_str, y_center))

            if not detected_numbers:
                return [(str(fallback_start_id + i), None) for i in range(total_structures)]

            # Sort by Y position (top to bottom)
            detected_numbers.sort(key=lambda x: x[1])

            # If we found more numbers than structures, try to pick the best subset
            # If fewer, we'll use fallback for unmatched structures
            # Estimate Y position for each structure (evenly distributed)
            struct_y_positions = []
            if total_structures == 1:
                struct_y_positions = [height // 2]
            else:
                margin = height * 0.1
                usable_height = height - 2 * margin
                for i in range(total_structures):
                    y = margin + (usable_height * i) / (total_structures - 1)
                    struct_y_positions.append(int(y))

            # Match each structure to the closest example number by Y position
            compound_results = []
            used_indices = set()

            for struct_idx in range(total_structures):
                struct_y = struct_y_positions[struct_idx]

                best_match = None
                best_distance = float('inf')
                best_idx = -1

                for num_idx, (num_str, num_y) in enumerate(detected_numbers):
                    if num_idx in used_indices:
                        continue
                    distance = abs(struct_y - num_y)
                    if distance < best_distance:
                        best_distance = distance
                        best_match = num_str
                        best_idx = num_idx

                if best_match is not None and best_distance < height * 0.4:
                    compound_results.append((best_match, None))
                    used_indices.add(best_idx)
                else:
                    compound_results.append((str(fallback_start_id + struct_idx), None))

            return compound_results

        except Exception:
            return [(str(fallback_start_id + i), None) for i in range(total_structures)]

    def _classify_structures(
        self,
        page_image,
        structure_boxes: list,
    ) -> Optional[list]:
        """Classify structures as example_compound or other using Claude Haiku.

        Returns None if the classifier is not available (no API key).
        """
        # Only classify if we have bounding boxes and the classifier is available
        valid_boxes = [box for box in structure_boxes if box is not None]
        if not valid_boxes or len(valid_boxes) != len(structure_boxes):
            return None

        try:
            from ..core.llm_compound_classifier import is_available, LLMCompoundClassifier
            if not is_available():
                return None

            classifier = LLMCompoundClassifier()
            return classifier.classify_page_structures(page_image, valid_boxes)
        except Exception as e:
            logger.debug("Compound classification failed: %s", e)
            return None

    @staticmethod
    def _format_eta(seconds: float) -> str:
        """Format seconds into a human-readable ETA string."""
        seconds = max(0, int(seconds))
        if seconds < 60:
            return f"~{seconds} sec remaining"
        minutes, secs = divmod(seconds, 60)
        if minutes < 60:
            return f"~{minutes} min {secs} sec remaining"
        hours, minutes = divmod(minutes, 60)
        return f"~{hours} hr {minutes} min remaining"

    def _emit_progress(
        self,
        current_page: int,
        total_pages: int,
        current_structure: int,
        total_structures: int,
        message: str
    ) -> None:
        """Emit a progress update signal."""
        progress = ProcessingProgress(
            current_page=current_page,
            total_pages=total_pages,
            current_structure=current_structure,
            total_structures=total_structures,
            status_message=message
        )
        self.progress_updated.emit(progress)

    @staticmethod
    def _normalize_compound_id_for_merge(raw_id: str) -> str:
        """Normalize a compound ID for merge matching.

        Strips common prefixes like "Example", "Compound", "Cpd.", "Ex." and
        removes leading zeros so that "007" matches "7".
        """
        import re
        normalized = raw_id.strip()
        # Strip common prefixes
        normalized = re.sub(
            r'^(?:Example|Compound|Cpd\.?|Cmpd\.?|Ex\.?|No\.?|#)\s*',
            '', normalized, flags=re.IGNORECASE,
        ).strip()
        # Strip leading zeros from purely numeric IDs: "007" -> "7"
        # But preserve compound IDs like "007a" or "00-7"
        m = re.match(r'^0+(\d+)$', normalized)
        if m:
            normalized = m.group(1)
        return normalized

    def _merge_biological_data(
        self,
        results: List[ExtractionResult],
        bio_data: dict,
        source_file: str
    ) -> int:
        """Merge biological data into extraction results by compound ID.

        Args:
            results: List of ExtractionResult to update.
            bio_data: Dictionary mapping compound_id to BiologicalData.
            source_file: Source file name to filter results.

        Returns:
            Number of compounds successfully merged.
        """
        if not bio_data:
            return 0

        import re

        merge_count = 0

        # Add merge debug to extractor's debug info for visibility
        debug_info = []
        debug_info.append(f"\n=== MERGE DEBUG ===")
        debug_info.append(f"Source file: {source_file}")
        debug_info.append(f"Bio data compounds: {list(bio_data.keys())}")

        # Get result compound IDs for this file
        result_ids = []
        for r in results:
            if r.compound_id:
                result_ids.append(f"{r.compound_id}(src={r.source_file})")
        debug_info.append(f"Result compound IDs: {result_ids[:15]}")

        # Build normalized lookup for bio_data keys
        # Maps normalized_id -> original_key for faster matching
        bio_normalized = {}
        for key in bio_data:
            norm = self._normalize_compound_id_for_merge(key)
            bio_normalized[norm] = key
        bio_data_keys = set(bio_data.keys())

        for result in results:
            # Only merge for results from this file
            if result.source_file != source_file:
                continue

            if not result.compound_id:
                continue

            # Normalize compound ID for matching
            compound_id = result.compound_id.strip()
            norm_compound_id = self._normalize_compound_id_for_merge(compound_id)
            matched_key = None

            # Try exact match first
            if compound_id in bio_data:
                matched_key = compound_id
            # Try normalized match (handles "Example 7" vs "7", leading zeros, etc.)
            elif norm_compound_id in bio_normalized:
                matched_key = bio_normalized[norm_compound_id]
            else:
                # Try case-insensitive exact match
                for key in bio_data_keys:
                    if key.lower() == compound_id.lower():
                        matched_key = key
                        break

                # Try case-insensitive normalized match
                if not matched_key:
                    norm_lower = norm_compound_id.lower()
                    for norm_key, orig_key in bio_normalized.items():
                        if norm_key.lower() == norm_lower:
                            matched_key = orig_key
                            break

                # Try matching just the numeric/alphanumeric part
                if not matched_key:
                    # Extract full ID including dash: "100-7" -> "100-7", "12a" -> "12a"
                    match = re.search(r'(\d+(?:-\d+)?[a-zA-Z]?)', norm_compound_id)
                    if match:
                        simple_id = match.group(1)
                        if simple_id in bio_data:
                            matched_key = simple_id
                        elif simple_id in bio_normalized:
                            matched_key = bio_normalized[simple_id]
                        else:
                            # Fuzzy match: compare extracted numeric parts
                            for key in bio_data_keys:
                                key_norm = self._normalize_compound_id_for_merge(key)
                                key_simple = re.search(r'(\d+(?:-\d+)?[a-zA-Z]?)', key_norm)
                                if key_simple and key_simple.group(1) == simple_id:
                                    matched_key = key
                                    break

                            # Last resort: compare just the numeric base
                            if not matched_key:
                                base_match = re.match(r'(\d+)', norm_compound_id)
                                if base_match:
                                    base_num = base_match.group(1).lstrip('0') or '0'
                                    for key in bio_data_keys:
                                        key_norm = self._normalize_compound_id_for_merge(key)
                                        key_base = re.match(r'(\d+)', key_norm)
                                        if key_base:
                                            key_base_num = key_base.group(1).lstrip('0') or '0'
                                            if key_base_num == base_num and '-' not in key_norm and '-' not in norm_compound_id:
                                                matched_key = key
                                                break

            if matched_key:
                data = bio_data[matched_key]
                # Legacy fields for backwards compatibility
                result.ic50 = data.ic50
                result.ec50 = data.ec50
                result.ki = data.ki
                result.kd = data.kd

                # Copy all other_assays data to result.bio_data for dynamic columns
                # This uses the actual header text as keys (e.g., "WiDr GI50 (nM)")
                for assay_name, value in data.other_assays.items():
                    result.bio_data[assay_name] = value

                merge_count += 1
                debug_info.append(f"  MATCHED: {compound_id} -> {matched_key}: {list(data.other_assays.items())[:2]}")

                # Mark as matched
                data.matched = True
            else:
                debug_info.append(f"  NOT MATCHED: {compound_id} (source={result.source_file})")

        # Log unmatched bio data compounds (extracted but not matched to structures)
        unmatched_bio = []
        for key, bd in bio_data.items():
            if not bd.matched:
                assays = list(bd.other_assays.items())[:3]
                unmatched_bio.append(f"{key}: {assays}")
        if unmatched_bio:
            debug_info.append(f"  UNMATCHED BIO DATA ({len(unmatched_bio)} compounds):")
            for entry in unmatched_bio[:20]:
                debug_info.append(f"    {entry}")

        # Add debug info to the extractor's log
        debug_info.append(f"Total merged: {merge_count}")
        self._bio_data_extractor._debug_info.extend(debug_info)

        return merge_count
