"""PDF processing module for extracting page images.

Uses pypdfium2 (Apache 2.0) for rendering and pdfplumber (MIT) for text extraction.
These are permissive-licensed alternatives to PyMuPDF (AGPL).
"""

from pathlib import Path
from typing import Generator, List, Optional
import pypdfium2 as pdfium
import pdfplumber
from PIL import Image

from ..models.extraction_result import PDFInfo


class PDFProcessor:
    """Handles PDF file operations and image extraction.

    Extracts high-resolution images from PDF pages for chemical structure detection.
    """

    DEFAULT_DPI = 200  # Balanced DPI for speed vs quality

    def __init__(self, dpi: int = DEFAULT_DPI):
        """Initialize the PDF processor.

        Args:
            dpi: Resolution for rendering PDF pages. Higher values give better
                 quality but slower processing. Default is 200 DPI.
        """
        self.dpi = dpi
        self._document: Optional[pdfium.PdfDocument] = None
        self._plumber_doc: Optional[pdfplumber.PDF] = None
        self._file_path: Optional[str] = None
        self._page_count: int = 0

    def open(self, file_path: str) -> PDFInfo:
        """Open a PDF file and return its information.

        Args:
            file_path: Path to the PDF file.

        Returns:
            PDFInfo object with file metadata.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file is not a valid PDF.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        try:
            # Open with pypdfium2 for rendering
            self._document = pdfium.PdfDocument(file_path)
            self._page_count = len(self._document)

            # Open with pdfplumber for text extraction
            self._plumber_doc = pdfplumber.open(file_path)

            self._file_path = file_path
        except Exception as e:
            raise ValueError(f"Failed to open PDF: {e}") from e

        if self._page_count == 0:
            raise ValueError("PDF has no pages")

        return PDFInfo(
            file_path=file_path,
            page_count=self._page_count
        )

    def close(self) -> None:
        """Close the currently open PDF document."""
        if self._document:
            self._document.close()
            self._document = None
        if self._plumber_doc:
            self._plumber_doc.close()
            self._plumber_doc = None
        self._file_path = None
        self._page_count = 0

    @property
    def is_open(self) -> bool:
        """Check if a PDF document is currently open."""
        return self._document is not None

    @property
    def page_count(self) -> int:
        """Get the number of pages in the open document."""
        return self._page_count

    def get_page_image(self, page_number: int) -> Image.Image:
        """Render a PDF page as a PIL Image.

        Args:
            page_number: 1-indexed page number.

        Returns:
            PIL Image of the rendered page.

        Raises:
            RuntimeError: If no document is open.
            ValueError: If page number is out of range.
        """
        if not self._document:
            raise RuntimeError("No PDF document is open")

        if page_number < 1 or page_number > self._page_count:
            raise ValueError(
                f"Page number {page_number} out of range "
                f"(1-{self._page_count})"
            )

        # pypdfium2 uses 0-indexed pages
        page = self._document[page_number - 1]

        # Calculate scale for desired DPI (72 DPI is PDF default)
        scale = self.dpi / 72.0

        # Render page to PIL Image
        bitmap = page.render(scale=scale)
        image = bitmap.to_pil()

        return image.convert("RGB")

    def iter_pages(self) -> Generator[tuple[int, Image.Image], None, None]:
        """Iterate over all pages in the document.

        Yields:
            Tuples of (page_number, page_image) where page_number is 1-indexed.

        Raises:
            RuntimeError: If no document is open.
        """
        if not self._document:
            raise RuntimeError("No PDF document is open")

        for page_num in range(1, self._page_count + 1):
            yield page_num, self.get_page_image(page_num)

    def get_page_text_blocks(self, page_number: int) -> List[dict]:
        """Extract text blocks with their bounding boxes from a page.

        Args:
            page_number: 1-indexed page number.

        Returns:
            List of dicts with keys: 'text', 'bbox' (x0, y0, x1, y1) in points.
            Coordinates are in PDF points (72 DPI), not image pixels.

        Raises:
            RuntimeError: If no document is open.
            ValueError: If page number is out of range.
        """
        if not self._plumber_doc:
            raise RuntimeError("No PDF document is open")

        if page_number < 1 or page_number > self._page_count:
            raise ValueError(
                f"Page number {page_number} out of range "
                f"(1-{self._page_count})"
            )

        # pdfplumber uses 0-indexed pages
        page = self._plumber_doc.pages[page_number - 1]

        text_blocks = []

        # Extract words with bounding boxes
        words = page.extract_words()
        for word in words:
            text_blocks.append({
                'text': word['text'],
                'bbox': (word['x0'], word['top'], word['x1'], word['bottom'])
            })

        return text_blocks

    def get_page_dimensions(self, page_number: int) -> tuple:
        """Get the dimensions of a page in PDF points.

        Args:
            page_number: 1-indexed page number.

        Returns:
            Tuple of (width, height) in PDF points (72 DPI).

        Raises:
            RuntimeError: If no document is open.
            ValueError: If page number is out of range.
        """
        if not self._document:
            raise RuntimeError("No PDF document is open")

        if page_number < 1 or page_number > self._page_count:
            raise ValueError(
                f"Page number {page_number} out of range "
                f"(1-{self._page_count})"
            )

        page = self._document[page_number - 1]
        return (page.get_width(), page.get_height())

    def __enter__(self) -> "PDFProcessor":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - ensures document is closed."""
        self.close()
