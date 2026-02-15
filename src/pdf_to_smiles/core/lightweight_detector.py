"""Lightweight chemical structure detection using OpenCV contour analysis.

No ML model required — uses adaptive thresholding and contour filtering
to detect chemical structure regions on a page. Pure OpenCV (already a dependency).
"""

from typing import List

import cv2
import numpy as np
from PIL import Image


class LightweightDetector:
    """Detects and segments chemical structures using OpenCV contours.

    Same interface as StructureDetector but requires no model download.
    Works by finding rectangular-ish dark regions of sufficient size.
    """

    MIN_STRUCTURE_SIZE = 50   # Minimum width/height in pixels
    MAX_ASPECT_RATIO = 5.0    # Filter elongated shapes (likely text)
    MIN_AREA = 5000           # Minimum contour area to filter noise

    # Morphological kernel size for closing gaps in structures
    MORPH_KERNEL_SIZE = 15
    # Padding around detected regions (pixels)
    CROP_PADDING = 10

    def detect_structures(self, page_image: Image.Image) -> List[Image.Image]:
        """Detect and extract chemical structures from a page image.

        Args:
            page_image: PIL Image of a PDF page.

        Returns:
            List of PIL Images, each containing a detected chemical structure.
        """
        try:
            img_array = np.array(page_image.convert('RGB'))
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

            # Adaptive threshold to handle varying background brightness
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, blockSize=25, C=10
            )

            # Morphological close to merge nearby strokes into blobs
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (self.MORPH_KERNEL_SIZE, self.MORPH_KERNEL_SIZE)
            )
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            # Find contours
            contours, _ = cv2.findContours(
                closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Filter and extract structure regions
            structure_images = []
            height, width = gray.shape

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.MIN_AREA:
                    continue

                x, y, w, h = cv2.boundingRect(contour)

                # Filter by minimum size
                if w < self.MIN_STRUCTURE_SIZE or h < self.MIN_STRUCTURE_SIZE:
                    continue

                # Filter by aspect ratio
                aspect_ratio = max(w, h) / max(min(w, h), 1)
                if aspect_ratio > self.MAX_ASPECT_RATIO:
                    continue

                # Filter by solidity (filled area vs convex hull area)
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0:
                    solidity = area / hull_area
                    # Chemical structures typically have solidity 0.1-0.8
                    # Very high solidity (>0.9) is likely a solid block/image
                    # Very low solidity (<0.05) is likely scattered noise
                    if solidity < 0.05:
                        continue

                # Crop with padding
                x1 = max(0, x - self.CROP_PADDING)
                y1 = max(0, y - self.CROP_PADDING)
                x2 = min(width, x + w + self.CROP_PADDING)
                y2 = min(height, y + h + self.CROP_PADDING)

                crop = img_array[y1:y2, x1:x2]
                structure_images.append(Image.fromarray(crop))

            # Sort by position (top-to-bottom, left-to-right)
            # Re-detect bounding boxes for sorting
            if len(structure_images) > 1:
                boxes_and_images = []
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < self.MIN_AREA:
                        continue
                    x, y, w, h = cv2.boundingRect(contour)
                    if w < self.MIN_STRUCTURE_SIZE or h < self.MIN_STRUCTURE_SIZE:
                        continue
                    aspect_ratio = max(w, h) / max(min(w, h), 1)
                    if aspect_ratio > self.MAX_ASPECT_RATIO:
                        continue
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    if hull_area > 0 and (area / hull_area) < 0.05:
                        continue
                    x1 = max(0, x - self.CROP_PADDING)
                    y1 = max(0, y - self.CROP_PADDING)
                    x2 = min(width, x + w + self.CROP_PADDING)
                    y2 = min(height, y + h + self.CROP_PADDING)
                    crop = img_array[y1:y2, x1:x2]
                    boxes_and_images.append((y1, x1, Image.fromarray(crop)))

                boxes_and_images.sort(key=lambda item: (item[0], item[1]))
                structure_images = [img for _, _, img in boxes_and_images]

            return structure_images

        except Exception as e:
            raise RuntimeError(f"Lightweight structure detection failed: {e}") from e
