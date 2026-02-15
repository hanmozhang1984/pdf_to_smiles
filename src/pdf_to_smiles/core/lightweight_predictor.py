"""Lightweight SMILES prediction using MolScribe (PyTorch CPU).

MolScribe is a transformer-based model for chemical structure recognition.
~400MB model, downloaded from HuggingFace Hub on first use and cached to
~/.cache/huggingface/. Uses PyTorch CPU — no GPU required.
"""

from __future__ import annotations

from typing import Optional, List

from PIL import Image


class LightweightPredictor:
    """Predicts SMILES strings from chemical structure images using MolScribe.

    Same interface as SMILESPredictor but uses MolScribe (PyTorch) instead
    of DECIMER (TensorFlow). Lighter weight and faster on CPU.
    """

    def __init__(self):
        self._model = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy initialization — downloads model on first use."""
        if self._initialized:
            return

        try:
            import torch
            from molscribe import MolScribe

            device = torch.device('cpu')
            self._model = MolScribe(device=device)
            self._initialized = True
        except ImportError as e:
            raise RuntimeError(
                "MolScribe is not installed. Install with:\n"
                "  pip install molscribe torch\n"
                f"Original error: {e}"
            ) from e

    @property
    def is_initialized(self) -> bool:
        """Check if the model has been loaded."""
        return self._initialized

    def predict(
        self, structure_image: Image.Image, high_accuracy: bool = False
    ) -> Optional[str]:
        """Predict SMILES string from a chemical structure image.

        Args:
            structure_image: PIL Image containing a single chemical structure.
            high_accuracy: If True, apply preprocessing for better accuracy.

        Returns:
            Predicted SMILES string, or None if prediction fails.
        """
        self._ensure_initialized()

        try:
            # MolScribe expects RGB PIL Image
            if structure_image.mode != 'RGB':
                structure_image = structure_image.convert('RGB')

            output = self._model.predict_image(structure_image)

            if isinstance(output, dict):
                smiles = output.get('smiles', '')
            elif isinstance(output, str):
                smiles = output
            else:
                return None

            if smiles and isinstance(smiles, str) and len(smiles.strip()) > 0:
                result = smiles.strip()
                # Validate with RDKit if available
                if high_accuracy and not self._is_valid_smiles(result):
                    return None
                return result
            return None

        except Exception as e:
            print(f"MolScribe prediction failed: {e}")
            return None

    def predict_batch(
        self,
        structure_images: List[Image.Image],
        high_accuracy: bool = False
    ) -> List[Optional[str]]:
        """Predict SMILES for multiple structure images.

        Args:
            structure_images: List of PIL Images containing chemical structures.
            high_accuracy: If True, apply preprocessing for better accuracy.

        Returns:
            List of predicted SMILES strings (None for failed predictions).
        """
        self._ensure_initialized()

        try:
            # Ensure all images are RGB
            rgb_images = []
            for img in structure_images:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                rgb_images.append(img)

            outputs = self._model.predict_images(rgb_images)

            results = []
            for output in outputs:
                if isinstance(output, dict):
                    smiles = output.get('smiles', '')
                elif isinstance(output, str):
                    smiles = output
                else:
                    results.append(None)
                    continue

                if smiles and isinstance(smiles, str) and len(smiles.strip()) > 0:
                    result = smiles.strip()
                    if high_accuracy and not self._is_valid_smiles(result):
                        results.append(None)
                    else:
                        results.append(result)
                else:
                    results.append(None)

            return results

        except Exception:
            # Fall back to one-by-one prediction
            return [self.predict(img, high_accuracy) for img in structure_images]

    @staticmethod
    def _is_valid_smiles(smiles: str) -> bool:
        """Quick check if SMILES is likely valid using RDKit."""
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except Exception:
            return False
