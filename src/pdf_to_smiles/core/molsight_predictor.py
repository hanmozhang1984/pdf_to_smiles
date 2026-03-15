"""MolSight SMILES predictor running in an isolated venv subprocess.

MolSight (hustvl/MolSight) uses an EfficientViT encoder + transformer decoder
for chemical structure recognition. It runs in ~/Documents/Projects/MolSight/venv to avoid
dependency conflicts with the main app.

Improvements over raw MolSight:
1. Image preprocessing — clean_structure_image + text masking before prediction
2. SCF3 post-processing — fix known invalid sulfur valence patterns
3. Multi-variant retry — CLAHE / adaptive threshold on failure
4. MolScribe fallback — try MolScribe when MolSight returns invalid SMILES
"""

from __future__ import annotations

import logging
import os
import re
import textwrap
from typing import List, Optional

from PIL import Image

from .subprocess_predictor import SubprocessPredictor

logger = logging.getLogger(__name__)


def _fix_sulfur_valence(smiles: str) -> str:
    """Fix known invalid sulfur valence patterns from MolSight output.

    MolSight misreads SCF3 (trifluoromethylthio) in several ways:
      - S(F)(F)(F)(F)F → SC(F)(F)F   (SF5: pentafluorosulfanyl)
      - S(F)(F)F       → SC(F)(F)F   (SF3: trifluorosulfanyl)
      - [SH](F)(F)F    → SC(F)(F)F   (protonated sulfur + 3F)
      - [SH]=C(F)F     → SC(F)(F)F   (S=C instead of S-C)
      - [SH]=*(F)F     → SC(F)(F)F   (wildcard carbon)
      - [SH]=[C-](F)(F)F → SC(F)(F)F (carbanion)

    Only applies the fix if the result is a valid SMILES.
    """
    if not smiles:
        return smiles

    patterns = [
        # SF5: S(F)(F)(F)(F)F → SC(F)(F)F
        (r'S\(F\)\(F\)\(F\)\(F\)F', 'SC(F)(F)F'),
        # [SH]=[C-](F)(F)F → SC(F)(F)F
        (r'\[SH\]=\[C-\]\(F\)\(F\)F', 'SC(F)(F)F'),
        # [SH]=C(F)F → SC(F)(F)F
        (r'\[SH\]=C\(F\)F', 'SC(F)(F)F'),
        # [SH]=*(F)F → SC(F)(F)F
        (r'\[SH\]=\*\(F\)F', 'SC(F)(F)F'),
        # [SH](F)(F)F → SC(F)(F)F
        (r'\[SH\]\(F\)\(F\)F', 'SC(F)(F)F'),
        # S(F)(F)F — bare SF3 (not part of S(=O) sulfonyl)
        (r'(?<!=\))S\(F\)\(F\)F', 'SC(F)(F)F'),
    ]

    modified = smiles
    for pattern, replacement in patterns:
        modified = re.sub(pattern, replacement, modified)

    if modified == smiles:
        return smiles

    # Validate the fix
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(modified)
        if mol is not None:
            return Chem.CanonSmiles(modified)
    except Exception:
        pass

    return smiles  # Revert if fix produces invalid SMILES


def _is_valid_smiles(smiles: str) -> bool:
    """Check if SMILES is valid using RDKit."""
    if not smiles or smiles == "NONE":
        return False
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        if mol.GetNumAtoms() < 3:
            return False
        return True
    except Exception:
        return False


MOLSIGHT_DIR = os.path.join(os.path.expanduser("~"), "Documents", "Projects", "MolSight")


class MolSightPredictor(SubprocessPredictor):
    VENV_PATH = os.path.join(MOLSIGHT_DIR, "venv")

    # Template uses {checkpoint_path} and {molsight_dir} placeholders, filled in __init__
    _WORKER_SCRIPT_TEMPLATE = textwrap.dedent("""\
        import sys
        import os

        # Run from MolSight directory so relative paths (vocab/, checkpoints) work
        os.chdir("{molsight_dir}")
        sys.path.insert(0, "{molsight_dir}")

        import cv2
        import torch
        import torch.nn.functional as F
        from types import SimpleNamespace

        from molsight.dataset import get_transforms
        from molsight.model import MolsightModel, get_edge_prediction
        from molsight.tokenizer import CharTokenizer, PAD_ID
        from molsight.chemistry import _postprocess_smiles

        try:
            from rdkit import Chem
        except ImportError:
            Chem = None

        # Build args matching inference.py defaults
        args = SimpleNamespace(
            encoder="efficientvit",
            use_checkpoint=False,
            embed_dim=512,
            dec_n_layer=6,
            dec_n_head=8,
            use_qknorm=True,
            use_swiglu=True,
            use_rmsnorm=True,
            lora={use_lora},
            regression=False,
            input_size=512,
            formats=["char", "edges"],
            vocab_file="vocab/vocab_chars.json",
            resume=True,
            max_len=320,
            beam_size=1,
            n_samples=1,
            save_attns=False,
            molblock=False,
            compute_confidence=False,
            keep_main_molecule=False,
        )

        tokenizer = CharTokenizer(args.vocab_file)
        model = MolsightModel(args, tokenizer)
        device = "cpu"
        model.to(device)

        # Load checkpoint
        ckpt_path = "{checkpoint_path}"
        if not os.path.exists(ckpt_path):
            import urllib.request
            url = "https://huggingface.co/Robert-zwr/MolSight/resolve/main/pubchem_uspto_smiles_edges_30.pth?download=true"
            print(f"Downloading MolSight weights...", file=sys.stderr, flush=True)
            urllib.request.urlretrieve(url, ckpt_path)

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state = checkpoint["model"]
        state = {{(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}}
        model.load_state_dict(state, strict=False)

        if {use_lora}:
            from molsight.model import enable_lora
            enable_lora(model)

        if hasattr(model, "module"):
            model = model.module
        model.eval()

        transform = get_transforms(args, augment=False, rotate=False)

        print("READY", flush=True)

        while True:
            line = sys.stdin.readline()
            if not line:
                break
            path = line.strip()
            if not path:
                continue
            try:
                image = cv2.imread(path)
                if image is None:
                    print("NONE", flush=True)
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                augmented = transform(image=image)
                tensor = augmented["image"].unsqueeze(0).to(device)

                with torch.no_grad():
                    kv_cache, hooks = model.install_kv_cache_hooks()
                    batch_preds, inter = model.generate(image=tensor, kv_cache=kv_cache)
                    for hook in hooks:
                        hook.remove()

                    # Edge prediction
                    if "edges" in args.formats:
                        hidden_states = inter["hidden_states"]
                        atom_indices = [torch.LongTensor(i) for i in batch_preds["indices"]]
                        atom_indices = torch.nn.utils.rnn.pad_sequence(
                            atom_indices, batch_first=True, padding_value=PAD_ID
                        ).to(hidden_states.device)
                        atom_indices = atom_indices + model.sample_begin
                        edge_logits = model.decoder.edge_predictor(hidden_states, atom_indices)
                        edge_probs = F.softmax(edge_logits, dim=-1)
                        valid_lengths = [len(ind) for ind in batch_preds["indices"]]
                        edge_preds, _ = get_edge_prediction(edge_probs, valid_lengths)
                        batch_preds["edges"] = edge_preds

                raw_smiles = batch_preds["smiles"][0]
                smiles, _, success = _postprocess_smiles(raw_smiles)

                if Chem is not None:
                    try:
                        smiles = Chem.CanonSmiles(smiles)
                    except Exception:
                        pass

                if smiles and len(smiles.strip()) > 0:
                    print(smiles.strip(), flush=True)
                else:
                    print("NONE", flush=True)
            except Exception as e:
                print(f"NONE", flush=True)
                print(f"Error: {{e}}", file=sys.stderr, flush=True)
    """)

    # Default WORKER_SCRIPT for backward compatibility (used by SubprocessPredictor)
    WORKER_SCRIPT = _WORKER_SCRIPT_TEMPLATE.format(
        molsight_dir=MOLSIGHT_DIR,
        checkpoint_path="pubchem_uspto_smiles_edges_30.pth",
        use_lora="False",
    )

    def __init__(self, checkpoint_path: str = "pubchem_uspto_smiles_edges_30.pth"):
        # Determine if this is a LoRA checkpoint (GRPO-trained)
        use_lora = "grpo" in checkpoint_path.lower() or "lora" in checkpoint_path.lower()
        # Set WORKER_SCRIPT before super().__init__() which starts the subprocess
        self.WORKER_SCRIPT = self._WORKER_SCRIPT_TEMPLATE.format(
            molsight_dir=MOLSIGHT_DIR,
            checkpoint_path=checkpoint_path,
            use_lora=str(use_lora),
        )
        super().__init__()
        self._molscribe_fallback = None
        # Check if text masking is available
        try:
            from pdf_to_smiles.core.text_masker import is_available as _text_masker_available
            self._mask_text = _text_masker_available()
        except Exception:
            self._mask_text = False

    def _get_molscribe_fallback(self):
        """Lazy-load MolScribe as fallback predictor."""
        if self._molscribe_fallback is None:
            from .lightweight_predictor import LightweightPredictor
            self._molscribe_fallback = LightweightPredictor()
        return self._molscribe_fallback

    def _preprocess(self, image: Image.Image) -> Image.Image:
        """Apply image cleaning + text masking before MolSight prediction."""
        from pdf_to_smiles.core.image_cleaner import clean_structure_image
        return clean_structure_image(image, mask_text=self._mask_text)

    @staticmethod
    def _enhance_image(image: Image.Image, aggressive: bool = False) -> Image.Image:
        """Enhance image for retry attempts (same as LightweightPredictor)."""
        import cv2
        import numpy as np

        if image.mode != 'RGB':
            image = image.convert('RGB')
        img = np.array(image)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        if aggressive:
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, blockSize=31, C=15
            )
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            thresh_val, _ = cv2.threshold(enhanced, 0, 255,
                                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary = np.where(enhanced > thresh_val, 255, enhanced).astype(np.uint8)
            ink_pixels = binary[binary < 255]
            if len(ink_pixels) > 0:
                lo, hi = np.percentile(ink_pixels, [2, 98])
                if hi > lo:
                    binary = np.clip((binary.astype(float) - lo) / (hi - lo) * 200, 0, 255).astype(np.uint8)
                    binary[binary > 200] = 255

        result = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(result)

    @staticmethod
    def _best_result(*candidates: Optional[str]) -> Optional[str]:
        """Pick the best SMILES from candidates: prefer connected over disconnected."""
        best_connected = None
        best_disconnected = None
        for s in candidates:
            if not _is_valid_smiles(s):
                continue
            if '.' not in s:
                if best_connected is None:
                    best_connected = s
            else:
                if best_disconnected is None:
                    best_disconnected = s
        return best_connected or best_disconnected

    def predict(
        self,
        image: Image.Image,
        high_accuracy: bool = False,
    ) -> Optional[str]:
        """Predict SMILES with multi-variant MolSight, post-processing, and fallback.

        Pipeline — raw image first (MolSight has its own transforms):
        1. Raw image → MolSight (best case, no interference)
        2. Cleaned image (contaminant removal + text mask) → MolSight
        3. CLAHE-enhanced cleaned image → MolSight
        4. If all MolSight attempts failed → MolScribe fallback

        At each step, apply SCF3 post-processing and prefer connected SMILES.
        """
        molsight_results: list[Optional[str]] = []

        # Step 1: Raw image — MolSight's internal transforms handle it best
        result = self.predict_single(image)
        result = _fix_sulfur_valence(result) if result else result
        molsight_results.append(result)

        # Early return if we got a valid connected SMILES
        if result and _is_valid_smiles(result) and '.' not in result:
            return result

        # Step 2: Cleaned image (remove bleeding lines, mask text)
        cleaned = self._preprocess(image)
        result = self.predict_single(cleaned)
        result = _fix_sulfur_valence(result) if result else result
        molsight_results.append(result)

        if result and _is_valid_smiles(result) and '.' not in result:
            return result

        # Step 3: CLAHE enhancement on cleaned image
        enhanced = self._enhance_image(cleaned, aggressive=False)
        result = self.predict_single(enhanced)
        result = _fix_sulfur_valence(result) if result else result
        molsight_results.append(result)

        # Check if any MolSight variant gave a usable result
        best = self._best_result(*molsight_results)
        if best is not None:
            return best

        # Step 4: MolScribe fallback — only if ALL MolSight variants failed
        try:
            fallback = self._get_molscribe_fallback()
            result = fallback.predict(image, high_accuracy=high_accuracy)
            if result:
                logger.info("MolScribe fallback succeeded where MolSight failed")
                return result
        except Exception as e:
            logger.warning("MolScribe fallback failed: %s", e)

        return None

    def predict_batch(
        self,
        images: List[Image.Image],
        high_accuracy: bool = False,
    ) -> List[Optional[str]]:
        """Predict SMILES for a batch with full improvement pipeline."""
        results = []
        for img in images:
            results.append(self.predict(img, high_accuracy=high_accuracy))
        return results

    def close(self) -> None:
        super().close()
        self._molscribe_fallback = None
