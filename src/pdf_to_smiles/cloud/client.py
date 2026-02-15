"""Cloud inference client for calling Modal.com backend."""

from __future__ import annotations

import base64
import io
from typing import Optional
from PIL import Image
import requests


class CloudInferenceClient:
    """Client for calling the cloud DECIMER inference API.

    This client communicates with the Modal.com deployed backend to perform
    GPU-accelerated structure detection and SMILES prediction.
    """

    # Default endpoint - deployed on Modal.com
    DEFAULT_ENDPOINT = "https://hanmozhang1984--pdf-to-smiles-process-image.modal.run"

    def __init__(self, endpoint: Optional[str] = None, timeout: int = 300):
        """Initialize the cloud inference client.

        Args:
            endpoint: Modal.com web endpoint URL. If None, uses DEFAULT_ENDPOINT.
            timeout: Request timeout in seconds (default 5 min for cold starts).
        """
        self.endpoint = endpoint or self.DEFAULT_ENDPOINT
        self.timeout = timeout
        self._session = requests.Session()

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buf = io.BytesIO()
        image.save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def _base64_to_image(self, b64: str) -> Image.Image:
        """Convert base64 string to PIL Image."""
        image_bytes = base64.b64decode(b64)
        return Image.open(io.BytesIO(image_bytes))

    def _call_api(self, action: str, **kwargs) -> dict:
        """Make API call to cloud backend.

        Args:
            action: API action (segment, predict, predict_batch)
            **kwargs: Additional request parameters

        Returns:
            Response dict with 'success', 'result', and optionally 'error'

        Raises:
            ConnectionError: If unable to reach the API
            RuntimeError: If API returns an error
        """
        payload = {"action": action, **kwargs}

        try:
            response = self._session.post(
                self.endpoint,
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            result = response.json()

            if not result.get("success"):
                raise RuntimeError(f"Cloud API error: {result.get('error', 'Unknown error')}")

            return result

        except requests.exceptions.Timeout:
            raise ConnectionError(f"Cloud API timeout after {self.timeout}s")
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(f"Unable to reach cloud API: {e}")
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"Cloud API HTTP error: {e}")

    def segment_structures(self, page_image: Image.Image) -> list[Image.Image]:
        """Detect and segment chemical structures from a page image.

        Args:
            page_image: PIL Image of a PDF page.

        Returns:
            List of PIL Images, each containing a detected chemical structure.
        """
        image_b64 = self._image_to_base64(page_image)
        result = self._call_api("segment", image=image_b64)

        # Convert base64 segments back to PIL Images
        segments = []
        for seg_b64 in result.get("result", []):
            segments.append(self._base64_to_image(seg_b64))

        return segments

    def predict_smiles(self, structure_image: Image.Image) -> Optional[str]:
        """Predict SMILES string from a chemical structure image.

        Args:
            structure_image: PIL Image containing a single chemical structure.

        Returns:
            Predicted SMILES string, or None if prediction fails.
        """
        image_b64 = self._image_to_base64(structure_image)
        result = self._call_api("predict", image=image_b64)
        return result.get("result")

    def predict_smiles_batch(self, structure_images: list[Image.Image]) -> list[Optional[str]]:
        """Predict SMILES for multiple structure images.

        Args:
            structure_images: List of PIL Images containing chemical structures.

        Returns:
            List of predicted SMILES strings (None for failed predictions).
        """
        images_b64 = [self._image_to_base64(img) for img in structure_images]
        result = self._call_api("predict_batch", images=images_b64)
        return result.get("result", [])

    def health_check(self) -> bool:
        """Check if the cloud API is reachable and responsive.

        Returns:
            True if API is healthy, False otherwise.
        """
        try:
            # Simple health check with minimal payload
            dummy_img = Image.new('RGB', (10, 10), 'white')
            image_b64 = self._image_to_base64(dummy_img)
            self._call_api("predict", image=image_b64)
            return True
        except Exception:
            return False

    def set_endpoint(self, endpoint: str) -> None:
        """Update the API endpoint URL.

        Args:
            endpoint: New Modal.com web endpoint URL.
        """
        self.endpoint = endpoint

    def close(self) -> None:
        """Close the HTTP session."""
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
