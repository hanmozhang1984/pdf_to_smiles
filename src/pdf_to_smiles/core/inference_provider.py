"""Unified inference provider that abstracts local vs cloud processing."""

from __future__ import annotations

from typing import Optional, List, Tuple
from PIL import Image

from .inference_settings import InferenceSettings, InferenceMode


class InferenceProvider:
    """Unified interface for structure detection and SMILES prediction.

    Automatically routes requests to local, lightweight, or cloud backends
    based on InferenceSettings configuration.

    Usage:
        provider = InferenceProvider()
        structures = provider.detect_structures(page_image)
        smiles_list = provider.predict_smiles_batch(structures)
    """

    def __init__(self):
        """Initialize the inference provider."""
        self._settings = InferenceSettings.get_instance()

        # Lazy-loaded backends
        self._local_detector = None
        self._local_predictor = None
        self._lightweight_detector = None
        self._lightweight_predictor = None
        self._cloud_client = None

    def _get_local_detector(self):
        """Get or create local structure detector (DECIMER)."""
        if self._local_detector is None:
            from .structure_detector import StructureDetector
            self._local_detector = StructureDetector()
        return self._local_detector

    def _get_local_predictor(self):
        """Get or create local SMILES predictor (DECIMER)."""
        if self._local_predictor is None:
            from .smiles_predictor import SMILESPredictor
            self._local_predictor = SMILESPredictor()
        return self._local_predictor

    def _get_lightweight_detector(self):
        """Get or create lightweight structure detector (OpenCV)."""
        if self._lightweight_detector is None:
            from .lightweight_detector import LightweightDetector
            self._lightweight_detector = LightweightDetector()
        return self._lightweight_detector

    def _get_lightweight_predictor(self):
        """Get or create lightweight SMILES predictor (MolScribe)."""
        if self._lightweight_predictor is None:
            from .lightweight_predictor import LightweightPredictor
            self._lightweight_predictor = LightweightPredictor()
        return self._lightweight_predictor

    def _get_cloud_client(self):
        """Get or create cloud inference client."""
        if self._cloud_client is None:
            from ..cloud import CloudInferenceClient
            self._cloud_client = CloudInferenceClient(
                endpoint=self._settings.cloud_endpoint,
                timeout=self._settings.cloud_timeout
            )
        return self._cloud_client

    def detect_structures(self, page_image: Image.Image) -> List[Image.Image]:
        """Detect and segment chemical structures from a page image.

        Args:
            page_image: PIL Image of a PDF page.

        Returns:
            List of PIL Images, each containing a detected chemical structure.
        """
        if self._settings.is_cloud:
            client = self._get_cloud_client()
            return client.segment_structures(page_image)
        elif self._settings.is_lightweight:
            detector = self._get_lightweight_detector()
            return detector.detect_structures(page_image)
        else:
            detector = self._get_local_detector()
            return detector.detect_structures(page_image)

    def detect_structures_with_boxes(
        self, page_image: Image.Image
    ) -> List[Tuple[Image.Image, Optional[Tuple[int, int, int, int]]]]:
        """Detect structures and return images with bounding boxes.

        Args:
            page_image: PIL Image of a PDF page.

        Returns:
            List of (cropped_image, (x1, y1, x2, y2)) tuples.
            Bounding box is None for backends that don't support it.
        """
        if self._settings.is_lightweight:
            detector = self._get_lightweight_detector()
            return detector.detect_structures_with_boxes(page_image)
        else:
            # Other backends don't support bounding boxes — return None for each
            structures = self.detect_structures(page_image)
            return [(img, None) for img in structures]

    def predict_smiles(
        self,
        structure_image: Image.Image,
        high_accuracy: bool = False
    ) -> Optional[str]:
        """Predict SMILES string from a chemical structure image.

        Args:
            structure_image: PIL Image containing a single chemical structure.
            high_accuracy: If True and using local inference, apply extra
                          preprocessing for complex structures.

        Returns:
            Predicted SMILES string, or None if prediction fails.
        """
        if self._settings.is_cloud:
            client = self._get_cloud_client()
            return client.predict_smiles(structure_image)
        elif self._settings.is_lightweight:
            predictor = self._get_lightweight_predictor()
            return predictor.predict(structure_image, high_accuracy=high_accuracy)
        else:
            predictor = self._get_local_predictor()
            return predictor.predict(structure_image, high_accuracy=high_accuracy)

    def predict_smiles_batch(
        self,
        structure_images: List[Image.Image],
        high_accuracy: bool = False
    ) -> List[Optional[str]]:
        """Predict SMILES for multiple structure images.

        Args:
            structure_images: List of PIL Images containing chemical structures.
            high_accuracy: If True and using local inference, apply extra
                          preprocessing for complex structures.

        Returns:
            List of predicted SMILES strings (None for failed predictions).
        """
        if self._settings.is_cloud:
            client = self._get_cloud_client()
            return client.predict_smiles_batch(structure_images)
        elif self._settings.is_lightweight:
            predictor = self._get_lightweight_predictor()
            return predictor.predict_batch(structure_images, high_accuracy=high_accuracy)
        else:
            predictor = self._get_local_predictor()
            return predictor.predict_batch(structure_images, high_accuracy=high_accuracy)

    def check_availability(self) -> tuple[bool, str]:
        """Check if the current inference backend is available.

        Returns:
            Tuple of (is_available, status_message)
        """
        if self._settings.is_cloud:
            try:
                client = self._get_cloud_client()
                if client.health_check():
                    return True, "Cloud GPU connected"
                else:
                    return False, "Cloud API not responding"
            except Exception as e:
                return False, f"Cloud connection failed: {e}"

        elif self._settings.is_lightweight:
            try:
                import molscribe  # noqa: F401
                import torch  # noqa: F401
                return True, "Lightweight mode ready (MolScribe + OpenCV)"
            except ImportError:
                return False, (
                    "MolScribe not installed. Install with:\n"
                    "pip install molscribe torch"
                )

        elif self._settings.is_local_gpu:
            try:
                import tensorflow as tf
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    return True, f"Local GPU ready ({len(gpus)} device(s))"
                else:
                    return False, "No GPU detected - install CUDA drivers"
            except Exception as e:
                return False, f"GPU check failed: {e}"

        else:  # CPU mode (DECIMER)
            try:
                import DECIMER  # noqa: F401
                return True, "CPU mode ready (DECIMER, processing will be slow)"
            except ImportError:
                return False, (
                    "DECIMER not installed. Install with:\n"
                    "pip install decimer decimer-segmentation tensorflow"
                )

    @property
    def mode(self) -> InferenceMode:
        """Current inference mode."""
        return self._settings.mode

    @mode.setter
    def mode(self, value: InferenceMode) -> None:
        """Set inference mode."""
        self._settings.mode = value
        # Reset cloud client if endpoint might have changed
        if value == InferenceMode.CLOUD:
            self._cloud_client = None

    def close(self) -> None:
        """Clean up resources."""
        if self._cloud_client is not None:
            self._cloud_client.close()
            self._cloud_client = None
