"""Lightweight SMILES prediction using MolScribe (PyTorch CPU).

MolScribe is a transformer-based model for chemical structure recognition.
~400MB model, downloaded from HuggingFace Hub on first use and cached to
~/.cache/huggingface/. Uses PyTorch CPU — no GPU required.
"""

from __future__ import annotations

from typing import Optional, List

import cv2
import numpy as np
from PIL import Image


class LightweightPredictor:
    """Predicts SMILES strings from chemical structure images using MolScribe.

    Same interface as SMILESPredictor but uses MolScribe (PyTorch) instead
    of DECIMER (TensorFlow). Lighter weight and faster on CPU.
    """

    def __init__(self, mask_text: bool | None = None):
        self._model = None
        self._initialized = False
        if mask_text is None:
            from pdf_to_smiles.core.text_masker import is_available as _text_masker_available
            self._mask_text = _text_masker_available()
        else:
            self._mask_text = mask_text

    # HuggingFace repo and default checkpoint for MolScribe
    _HF_REPO = "yujieq/MolScribe"
    _HF_FILENAME = "swin_base_char_aux_1m.pth"

    def _ensure_initialized(self) -> None:
        """Lazy initialization — downloads model on first use."""
        if self._initialized:
            return

        try:
            import torch
            from molscribe import MolScribe
            from huggingface_hub import hf_hub_download

            # Download checkpoint from HuggingFace Hub (cached after first download)
            model_path = hf_hub_download(self._HF_REPO, self._HF_FILENAME)

            device = torch.device('cpu')
            self._model = MolScribe(model_path, device=device)

            # Patch MolScribe bugs:
            # 1. BOND_TYPES missing key 0 → KeyError expanding CF3, NO2, etc.
            # 2. convert_graph_to_smiles uses multiprocessing.Pool which fails
            #    when run from non-file contexts (stdin, frozen apps, threads).
            # Fix: patch BOND_TYPES and replace with single-process version.
            try:
                from rdkit import Chem
                import molscribe.chemistry as msc
                import molscribe.interface as msi
                if 0 not in msc.BOND_TYPES:
                    msc.BOND_TYPES[0] = Chem.BondType.SINGLE

                def _convert_graph_to_smiles_single(coords, symbols, edges,
                                                     images=None, num_workers=1):
                    if images is None:
                        images_iter = [None] * len(coords)
                    else:
                        images_iter = images
                    results = [
                        msc._convert_graph_to_smiles(c, s, e, img)
                        for c, s, e, img in zip(coords, symbols, edges, images_iter)
                    ]
                    smiles_list, molblock_list, success = zip(*results)
                    r_success = np.mean(success)
                    return smiles_list, molblock_list, r_success

                # Patch both the module and the local binding in interface.py
                msc.convert_graph_to_smiles = _convert_graph_to_smiles_single
                msi.convert_graph_to_smiles = _convert_graph_to_smiles_single
            except Exception:
                pass

            self._initialized = True
        except ImportError as e:
            raise RuntimeError(
                "MolScribe is not installed. Install with:\n"
                "  pip install molscribe torch huggingface_hub\n"
                f"Original error: {e}"
            ) from e

    @property
    def is_initialized(self) -> bool:
        """Check if the model has been loaded."""
        return self._initialized

    @staticmethod
    def _enhance_for_prediction(image: Image.Image, aggressive: bool = False) -> Image.Image:
        """Enhance image contrast and clean background for better prediction.

        Args:
            image: Cleaned PIL Image (after image_cleaner).
            aggressive: If True, apply stronger preprocessing (used on retry).

        Returns:
            Enhanced PIL Image with improved contrast.
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
        img = np.array(image)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        if aggressive:
            # Adaptive thresholding to handle uneven backgrounds
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, blockSize=31, C=15
            )
        else:
            # CLAHE for local contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            # Whiten background: pixels above the Otsu threshold become pure white
            thresh_val, _ = cv2.threshold(enhanced, 0, 255,
                                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary = np.where(enhanced > thresh_val, 255, enhanced).astype(np.uint8)

            # Stretch remaining ink to full contrast range
            ink_pixels = binary[binary < 255]
            if len(ink_pixels) > 0:
                lo, hi = np.percentile(ink_pixels, [2, 98])
                if hi > lo:
                    binary = np.clip((binary.astype(float) - lo) / (hi - lo) * 200, 0, 255).astype(np.uint8)
                    binary[binary > 200] = 255

        # Convert back to RGB
        result = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(result)

    def _predict_single(self, img_array: np.ndarray) -> Optional[str]:
        """Run MolScribe on a numpy RGB array, validate with RDKit.

        Returns:
            Valid SMILES string, or None.
        """
        try:
            output = self._model.predict_image(img_array)

            if isinstance(output, dict):
                smiles = output.get('smiles', '')
            elif isinstance(output, str):
                smiles = output
            else:
                return None

            if smiles and isinstance(smiles, str) and len(smiles.strip()) > 0:
                result = smiles.strip()
                if not self._is_valid_smiles(result):
                    return None
                return result
            return None
        except Exception:
            return None

    @staticmethod
    def _is_connected(smiles: str) -> bool:
        """Check if SMILES represents a single connected molecule (no '.')."""
        return '.' not in smiles

    def _predict_best(self, images: list[np.ndarray]) -> Optional[str]:
        """Try multiple image variants and return the best SMILES.

        Preference order:
        1. Connected (no '.') SMILES from the earliest image variant
        2. Disconnected SMILES as fallback (from the earliest variant)
        """
        best_disconnected = None
        for img in images:
            result = self._predict_single(img)
            if result is None:
                continue
            if self._is_connected(result):
                return result  # Connected — accept immediately
            if best_disconnected is None:
                best_disconnected = result  # Keep first disconnected as fallback
        return best_disconnected

    def predict(
        self, structure_image: Image.Image, high_accuracy: bool = False
    ) -> Optional[str]:
        """Predict SMILES string from a chemical structure image.

        Tries multiple preprocessing variants and prefers connected SMILES
        (single molecule) over disconnected fragments.

        Args:
            structure_image: PIL Image containing a single chemical structure.
            high_accuracy: If True, apply preprocessing for better accuracy.

        Returns:
            Predicted SMILES string, or None if prediction fails.
        """
        from pdf_to_smiles.core.image_cleaner import clean_structure_image
        cleaned = clean_structure_image(structure_image, mask_text=self._mask_text)

        self._ensure_initialized()

        # Prepare all image variants
        variants = []

        # Variant 1: original cleaned (no enhancement) — safest for thin bonds
        orig = cleaned.convert('RGB') if cleaned.mode != 'RGB' else cleaned
        variants.append(np.array(orig))

        # Variant 2: mild enhancement (CLAHE + background whitening)
        enhanced = self._enhance_for_prediction(cleaned, aggressive=False)
        variants.append(np.array(enhanced.convert('RGB')))

        # Variant 3: aggressive preprocessing (adaptive threshold)
        aggressive = self._enhance_for_prediction(cleaned, aggressive=True)
        variants.append(np.array(aggressive.convert('RGB')))

        return self._predict_best(variants)

    def predict_batch(
        self,
        structure_images: List[Image.Image],
        high_accuracy: bool = False
    ) -> List[Optional[str]]:
        """Predict SMILES for multiple structure images.

        Uses batch prediction with original images first, then retries
        failures/disconnected results with enhanced variants.

        Args:
            structure_images: List of PIL Images containing chemical structures.
            high_accuracy: If True, apply preprocessing for better accuracy.

        Returns:
            List of predicted SMILES strings (None for failed predictions).
        """
        from pdf_to_smiles.core.image_cleaner import clean_structure_image
        cleaned_images = [clean_structure_image(img, mask_text=self._mask_text)
                          for img in structure_images]

        self._ensure_initialized()

        # First pass: batch predict with original cleaned images
        try:
            np_images = []
            for img in cleaned_images:
                rgb = img.convert('RGB') if img.mode != 'RGB' else img
                np_images.append(np.array(rgb))

            outputs = self._model.predict_images(np_images)

            results: List[Optional[str]] = []
            for output in outputs:
                smiles = None
                if isinstance(output, dict):
                    smiles = output.get('smiles', '')
                elif isinstance(output, str):
                    smiles = output

                if smiles and isinstance(smiles, str) and len(smiles.strip()) > 0:
                    result = smiles.strip()
                    if self._is_valid_smiles(result):
                        results.append(result)
                    else:
                        results.append(None)
                else:
                    results.append(None)

        except Exception:
            results = [None] * len(structure_images)

        # Retry failures and disconnected results with enhanced variants
        for i, (r, cleaned) in enumerate(zip(results, cleaned_images)):
            if r is None or not self._is_connected(r):
                # Build variant images (skip original, already tried)
                variants = []
                mild = self._enhance_for_prediction(cleaned, aggressive=False)
                variants.append(np.array(mild.convert('RGB')))
                aggr = self._enhance_for_prediction(cleaned, aggressive=True)
                variants.append(np.array(aggr.convert('RGB')))

                for img in variants:
                    candidate = self._predict_single(img)
                    if candidate is not None and self._is_connected(candidate):
                        results[i] = candidate
                        break
                else:
                    # No connected result found; keep best disconnected
                    if r is None:
                        for img in variants:
                            candidate = self._predict_single(img)
                            if candidate is not None:
                                results[i] = candidate
                                break

        return results

    @staticmethod
    def _is_valid_smiles(smiles: str) -> bool:
        """Check if SMILES is a valid, plausible chemical structure."""
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            # Reject wildcard-heavy fragments (from non-structure images)
            num_atoms = mol.GetNumAtoms()
            if num_atoms < 3:
                return False
            wildcards = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 0)
            if wildcards > num_atoms * 0.3:
                return False
            return True
        except Exception:
            return False
