"""Chemical structure detection using DECIMER-Segmentation."""

from typing import List
from PIL import Image
import numpy as np


class StructureDetector:
    """Detects and segments chemical structures from page images.

    Uses DECIMER-Segmentation (Mask R-CNN based) to detect and extract
    individual chemical structure images from a page.
    """

    # Minimum size for valid chemical structures (filters noise)
    MIN_STRUCTURE_SIZE = 50  # pixels
    MAX_ASPECT_RATIO = 5.0   # filter extremely elongated shapes (likely text)

    def __init__(self):
        """Initialize the structure detector.

        Note: The actual model loading happens lazily on first use to avoid
        slow startup times.
        """
        self._segment_chemical_structures = None
        self._initialized = False

    def _is_valid_structure(self, img: Image.Image) -> bool:
        """Check if image is likely a valid chemical structure (not noise/text).

        Args:
            img: PIL Image to check.

        Returns:
            True if image passes size/shape filters.
        """
        width, height = img.size

        # Filter too small images (likely noise)
        if width < self.MIN_STRUCTURE_SIZE or height < self.MIN_STRUCTURE_SIZE:
            return False

        # Filter extremely elongated shapes (likely text or lines)
        aspect_ratio = max(width, height) / max(min(width, height), 1)
        if aspect_ratio > self.MAX_ASPECT_RATIO:
            return False

        return True

    def _ensure_initialized(self) -> None:
        """Lazy initialization of DECIMER-Segmentation."""
        if not self._initialized:
            # Import here to defer the heavy TensorFlow loading
            from decimer_segmentation import segment_chemical_structures
            self._segment_chemical_structures = segment_chemical_structures
            self._initialized = True

    def detect_structures(self, page_image: Image.Image) -> List[Image.Image]:
        """Detect and extract chemical structures from a page image.

        Args:
            page_image: PIL Image of a PDF page.

        Returns:
            List of PIL Images, each containing a detected chemical structure.
            Returns empty list if no structures are detected.

        Raises:
            RuntimeError: If structure detection fails.
        """
        self._ensure_initialized()

        try:
            # Convert PIL Image to numpy array (DECIMER expects np.array)
            image_array = np.array(page_image)

            # Run segmentation - returns list of numpy arrays
            segments = self._segment_chemical_structures(image_array)

            # Convert numpy arrays back to PIL Images and filter invalid ones
            structure_images = []
            for segment in segments:
                if isinstance(segment, np.ndarray):
                    # Handle grayscale, RGB, and RGBA arrays
                    if segment.ndim == 2:
                        img = Image.fromarray(segment, mode='L')
                    elif segment.ndim == 3 and segment.shape[2] == 4:
                        img = Image.fromarray(segment, mode='RGBA')
                    elif segment.ndim == 3 and segment.shape[2] == 3:
                        img = Image.fromarray(segment, mode='RGB')
                    else:
                        img = Image.fromarray(segment)
                    img = img.convert('RGB')
                elif isinstance(segment, Image.Image):
                    img = segment.convert('RGB')
                else:
                    continue

                # Filter out noise/text before expensive SMILES prediction
                if self._is_valid_structure(img):
                    structure_images.append(img)

            return structure_images

        except Exception as e:
            raise RuntimeError(f"Structure detection failed: {e}") from e

    def detect_structures_with_boxes(
        self, page_image: Image.Image
    ) -> List[tuple[Image.Image, tuple[int, int, int, int]]]:
        """Detect structures and return both images and bounding boxes.

        Args:
            page_image: PIL Image of a PDF page.

        Returns:
            List of tuples (structure_image, bounding_box) where bounding_box
            is (x1, y1, x2, y2) in pixel coordinates.

        Note:
            This is a placeholder - DECIMER-Segmentation's default API doesn't
            expose bounding boxes directly. For now, returns images with None boxes.
        """
        structures = self.detect_structures(page_image)
        # Return with None boxes since we don't have access to coordinates
        return [(img, None) for img in structures]
