"""SMILES prediction from chemical structure images using DECIMER."""

from __future__ import annotations

from typing import Optional
from PIL import Image
import tempfile
import os
import numpy as np


class SMILESPredictor:
    """Predicts SMILES strings from chemical structure images.

    Uses the DECIMER deep learning model to convert structure images to SMILES.
    """

    def __init__(self):
        """Initialize the SMILES predictor.

        Note: Model loading is deferred to first prediction to avoid slow startup.
        The first prediction will take longer (~30-60s) due to model loading.
        """
        self._predict_smiles = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy initialization of DECIMER model."""
        if not self._initialized:
            # Import here to defer heavy TensorFlow loading
            from DECIMER import predict_SMILES
            self._predict_smiles = predict_SMILES
            self._initialized = True

    def predict(self, structure_image: Image.Image, high_accuracy: bool = False) -> Optional[str]:
        """Predict SMILES string from a chemical structure image.

        Args:
            structure_image: PIL Image containing a single chemical structure.
            high_accuracy: If True, apply image preprocessing for better accuracy
                          on complex structures (macrocycles, etc.).

        Returns:
            Predicted SMILES string, or None if prediction fails.

        Note:
            - First call will be slow (~30-60s) due to model loading
            - Subsequent calls are faster (~1-3s per structure)
            - Accuracy is ~96% for clean, well-rendered structures
            - Stereochemistry may not be correctly predicted
        """
        self._ensure_initialized()

        if high_accuracy:
            # Try multiple preprocessing strategies for complex structures
            return self._predict_with_retry(structure_image)
        else:
            return self._predict_single(structure_image)

    def _predict_single(self, structure_image: Image.Image) -> Optional[str]:
        """Single prediction attempt without preprocessing."""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                input_path = os.path.join(temp_dir, "structure.png")
                structure_image.save(input_path, "PNG")
                smiles = self._predict_smiles(input_path)

                if smiles and isinstance(smiles, str) and len(smiles.strip()) > 0:
                    return smiles.strip()
                return None
        except Exception as e:
            print(f"SMILES prediction failed: {e}")
            return None

    def _predict_with_retry(self, structure_image: Image.Image) -> Optional[str]:
        """Try multiple preprocessing strategies and return the best result."""
        # Check if this is a wide/elongated structure (like macrocycles)
        width, height = structure_image.size
        aspect_ratio = max(width, height) / max(min(width, height), 1)
        is_elongated = aspect_ratio > 2.0

        if is_elongated:
            # For elongated structures, try square padding first
            result = self._try_predict(self._preprocess_square_pad(structure_image))
            if result and self._is_valid_smiles(result):
                return result

        # Strategy 1: Enhanced preprocessing (CLAHE + sharpening)
        result = self._try_predict(self._preprocess_enhanced(structure_image))
        if result and self._is_valid_smiles(result):
            return result

        # Strategy 2: Simple preprocessing (original method)
        result = self._try_predict(self._preprocess_simple(structure_image))
        if result and self._is_valid_smiles(result):
            return result

        # Strategy 3: Binarized (high contrast black/white)
        result = self._try_predict(self._preprocess_binarized(structure_image))
        if result and self._is_valid_smiles(result):
            return result

        # Strategy 4: High resolution (scale up more aggressively)
        result = self._try_predict(self._preprocess_highres(structure_image))
        if result and self._is_valid_smiles(result):
            return result

        # Strategy 5: Original with just padding
        result = self._try_predict(self._preprocess_minimal(structure_image))
        if result:
            return result

        return None

    def _try_predict(self, image: Image.Image) -> Optional[str]:
        """Attempt prediction on a preprocessed image."""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                input_path = os.path.join(temp_dir, "structure.png")
                image.save(input_path, "PNG")
                smiles = self._predict_smiles(input_path)

                if smiles and isinstance(smiles, str) and len(smiles.strip()) > 0:
                    return smiles.strip()
                return None
        except Exception:
            return None

    def _is_valid_smiles(self, smiles: str) -> bool:
        """Quick check if SMILES is likely valid using RDKit."""
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except Exception:
            return False

    def _preprocess_enhanced(self, image: Image.Image) -> Image.Image:
        """Enhanced preprocessing with CLAHE and sharpening for complex structures."""
        import cv2

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        img_array = np.array(image)

        # Convert to LAB color space for CLAHE on lightness channel
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Apply CLAHE to lightness channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_channel)

        # Merge channels back
        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        img_array = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

        # Apply mild sharpening (less aggressive for complex structures)
        kernel = np.array([[0, -1, 0],
                          [-1,  5, -1],
                          [0, -1, 0]])
        img_array = cv2.filter2D(img_array, -1, kernel)

        # Ensure white background
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        white_mask = gray > 240
        img_array[white_mask] = [255, 255, 255]

        image = Image.fromarray(img_array)

        # Smart resize preserving aspect ratio - target longer side to 600px
        # but don't shrink if already reasonable size
        width, height = image.size
        max_dim = max(width, height)
        min_dim = min(width, height)

        if max_dim < 350:
            # Too small - scale up
            scale = 500 / max_dim
            new_size = (int(width * scale), int(height * scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        elif max_dim > 900:
            # Too large - scale down
            scale = 700 / max_dim
            new_size = (int(width * scale), int(height * scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        # Add padding
        padding = 30
        new_image = Image.new('RGB',
            (image.width + 2 * padding, image.height + 2 * padding),
            (255, 255, 255))
        new_image.paste(image, (padding, padding))

        return new_image

    def _preprocess_simple(self, image: Image.Image) -> Image.Image:
        """Simple preprocessing with contrast enhancement."""
        if image.mode != 'RGB':
            image = image.convert('RGB')

        img_array = np.array(image)

        # Ensure white background
        white_threshold = 240
        mask = np.all(img_array > white_threshold, axis=2)
        img_array[mask] = [255, 255, 255]

        # Darken non-white pixels for better contrast
        non_white_mask = ~mask
        img_array[non_white_mask] = np.clip(
            img_array[non_white_mask] * 0.85, 0, 255
        ).astype(np.uint8)

        image = Image.fromarray(img_array)

        # Resize if too small
        min_dim = min(image.size)
        if min_dim < 400:
            scale = 450 / min_dim
            new_size = (int(image.width * scale), int(image.height * scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        # Add padding
        padding = 20
        new_image = Image.new('RGB',
            (image.width + 2 * padding, image.height + 2 * padding),
            (255, 255, 255))
        new_image.paste(image, (padding, padding))

        return new_image

    def _preprocess_binarized(self, image: Image.Image) -> Image.Image:
        """Binarized preprocessing for maximum contrast."""
        import cv2

        if image.mode != 'RGB':
            image = image.convert('RGB')

        img_array = np.array(image)

        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Apply Otsu's binarization
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Convert back to RGB
        img_array = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        image = Image.fromarray(img_array)

        # Resize to optimal size
        max_dim = max(image.size)
        if max_dim < 400 or max_dim > 800:
            target = 512
            scale = target / max_dim
            new_size = (int(image.width * scale), int(image.height * scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        # Add padding
        padding = 30
        new_image = Image.new('RGB',
            (image.width + 2 * padding, image.height + 2 * padding),
            (255, 255, 255))
        new_image.paste(image, (padding, padding))

        return new_image

    def _preprocess_minimal(self, image: Image.Image) -> Image.Image:
        """Minimal preprocessing - just resize and pad."""
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize to optimal size
        max_dim = max(image.size)
        if max_dim < 400:
            scale = 500 / max_dim
            new_size = (int(image.width * scale), int(image.height * scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        # Add padding
        padding = 25
        new_image = Image.new('RGB',
            (image.width + 2 * padding, image.height + 2 * padding),
            (255, 255, 255))
        new_image.paste(image, (padding, padding))

        return new_image

    def _preprocess_square_pad(self, image: Image.Image) -> Image.Image:
        """Pad elongated structures to make them more square (helps DECIMER)."""
        import cv2

        if image.mode != 'RGB':
            image = image.convert('RGB')

        img_array = np.array(image)

        # Apply CLAHE for contrast
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_channel)
        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        img_array = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

        # Ensure white background
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        white_mask = gray > 240
        img_array[white_mask] = [255, 255, 255]

        image = Image.fromarray(img_array)
        width, height = image.size

        # Scale up if needed
        max_dim = max(width, height)
        if max_dim < 500:
            scale = 600 / max_dim
            width = int(width * scale)
            height = int(height * scale)
            image = image.resize((width, height), Image.Resampling.LANCZOS)

        # Make it more square by adding padding to the shorter dimension
        target_size = max(width, height)
        pad_x = (target_size - width) // 2
        pad_y = (target_size - height) // 2

        # Add extra margin
        margin = 40
        new_size = target_size + 2 * margin
        new_image = Image.new('RGB', (new_size, new_size), (255, 255, 255))
        new_image.paste(image, (pad_x + margin, pad_y + margin))

        return new_image

    def _preprocess_highres(self, image: Image.Image) -> Image.Image:
        """High resolution preprocessing - scale up aggressively."""
        import cv2

        if image.mode != 'RGB':
            image = image.convert('RGB')

        img_array = np.array(image)

        # Ensure white background first
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        white_mask = gray > 235
        img_array[white_mask] = [255, 255, 255]

        # Enhance contrast on non-white pixels
        non_white = ~white_mask
        img_array[non_white] = np.clip(img_array[non_white] * 0.8, 0, 255).astype(np.uint8)

        image = Image.fromarray(img_array)

        # Scale up to 800px on longest side
        width, height = image.size
        max_dim = max(width, height)
        if max_dim < 800:
            scale = 800 / max_dim
            new_size = (int(width * scale), int(height * scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        # Add padding
        padding = 50
        new_image = Image.new('RGB',
            (image.width + 2 * padding, image.height + 2 * padding),
            (255, 255, 255))
        new_image.paste(image, (padding, padding))

        return new_image

    def predict_batch(self, structure_images: list[Image.Image], high_accuracy: bool = False) -> list[Optional[str]]:
        """Predict SMILES for multiple structure images.

        Args:
            structure_images: List of PIL Images containing chemical structures.
            high_accuracy: If True, apply image preprocessing for better accuracy.

        Returns:
            List of predicted SMILES strings (None for failed predictions).
        """
        return [self.predict(img, high_accuracy) for img in structure_images]

    @property
    def is_initialized(self) -> bool:
        """Check if the model has been loaded."""
        return self._initialized
