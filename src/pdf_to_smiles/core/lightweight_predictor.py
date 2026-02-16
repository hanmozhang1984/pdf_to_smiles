"""Lightweight SMILES prediction using MolScribe (PyTorch CPU).

MolScribe is a transformer-based model for chemical structure recognition.
~400MB model, downloaded from HuggingFace Hub on first use and cached to
~/.cache/huggingface/. Uses PyTorch CPU — no GPU required.
"""

from __future__ import annotations

from typing import Optional, List

import numpy as np
from PIL import Image


class LightweightPredictor:
    """Predicts SMILES strings from chemical structure images using MolScribe.

    Same interface as SMILESPredictor but uses MolScribe (PyTorch) instead
    of DECIMER (TensorFlow). Lighter weight and faster on CPU.
    """

    def __init__(self):
        self._model = None
        self._initialized = False

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
            # MolScribe expects numpy array (RGB)
            if structure_image.mode != 'RGB':
                structure_image = structure_image.convert('RGB')
            img_array = np.array(structure_image)

            output = self._model.predict_image(img_array)

            if isinstance(output, dict):
                smiles = output.get('smiles', '')
            elif isinstance(output, str):
                smiles = output
            else:
                return None

            if smiles and isinstance(smiles, str) and len(smiles.strip()) > 0:
                result = smiles.strip()
                # Always validate with RDKit — rejects garbage from non-structure inputs
                if not self._is_valid_smiles(result):
                    return None
                return result
            return None

        except Exception as e:
            import traceback
            print(f"MolScribe prediction failed: {e}")
            traceback.print_exc()
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
            # MolScribe expects numpy arrays (RGB)
            np_images = []
            for img in structure_images:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                np_images.append(np.array(img))

            outputs = self._model.predict_images(np_images)

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
                    if not self._is_valid_smiles(result):
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
