"""MolSight SMILES predictor running in an isolated venv subprocess.

MolSight (hustvl/MolSight) uses an EfficientViT encoder + transformer decoder
for chemical structure recognition. It runs in ~/Documents/Projects/MolSight/venv to avoid
dependency conflicts with the main app.

Improvements over raw MolSight:
1. Image preprocessing — clean_structure_image + text masking before prediction
2. SCF3 post-processing — fix known invalid sulfur valence patterns
3. Multi-variant retry — CLAHE / adaptive threshold on failure
4. MolScribe fallback — try MolScribe when MolSight returns invalid SMILES
5. Deuterium wildcard repair — fix * atoms from D₃C / [2H] labels
6. Stereodescriptor masking — remove (R), (S) text that confuses the model
"""

from __future__ import annotations

import logging
import os
import re
import textwrap
from typing import List, Optional, Tuple

from PIL import Image

from .subprocess_predictor import SubprocessPredictor

logger = logging.getLogger(__name__)


def _keep_largest_fragment(smiles: str) -> str:
    """Keep only the largest fragment from a dot-separated SMILES.

    When crops include ink from adjacent table rows (e.g. an isopropyl
    group bleeding in from the row above), MolSight reads them as a
    separate molecule joined by '.'.  The real structure is always the
    longest fragment; the contamination is short (10-20 chars).

    Only strips fragments when the length ratio is >=3:1 to avoid
    discarding legitimate salts/counterions of similar size.
    """
    if not smiles or '.' not in smiles:
        return smiles

    fragments = smiles.split('.')
    if len(fragments) < 2:
        return smiles

    longest = max(fragments, key=len)
    second = max((f for f in fragments if f is not longest), key=len, default='')

    # Only strip if the longest fragment is >=3x the second-longest,
    # so we don't discard genuine counterions (e.g. CF3COOH salts)
    if len(longest) >= 3 * max(len(second), 1):
        return longest

    return smiles


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


def _fix_deuterium_wildcards(smiles: str) -> str:
    """Replace wildcard atoms (*) that likely represent deuterium labels.

    MolSight reads D₃C (trideuteromethyl) and similar deuterium labels
    as * (wildcard) because [2H] is absent from its training vocabulary.

    Tries CD3 replacement first; if that produces an invalid SMILES
    (e.g., multiple wildcards cause valence issues), tries replacing
    wildcards one at a time.

    Only applies when the result is a valid SMILES with more atoms.
    """
    if not smiles or '*' not in smiles:
        return smiles

    from rdkit import Chem

    cd3 = '[2H]C([2H])([2H])'

    # Try replacing all wildcards at once
    modified = smiles.replace('*', cd3)
    try:
        new_mol = Chem.MolFromSmiles(modified)
        if new_mol is not None and new_mol.GetNumAtoms() >= 3:
            return Chem.CanonSmiles(modified)
    except Exception:
        pass

    # If bulk replacement failed, try one wildcard at a time
    # (some multi-wildcard SMILES only work with partial replacement)
    best = smiles
    current = smiles
    for _ in range(smiles.count('*')):
        candidate = current.replace('*', cd3, 1)
        try:
            mol = Chem.MolFromSmiles(candidate)
            if mol is not None and mol.GetNumAtoms() >= 3:
                best = Chem.CanonSmiles(candidate)
                current = candidate
            else:
                break
        except Exception:
            break

    return best


def _mask_stereodescriptors(image: Image.Image) -> Image.Image:
    """Mask italic (R), (S), (E), (Z) stereodescriptor text labels.

    Patent structure images often include italic stereodescriptor labels
    like (R), (S) next to stereocenters. These confuse MolSight into
    misreading or truncating adjacent structural features.

    Strategy: find small clusters of ink pixels that match the size and
    density profile of stereodescriptor labels, then white-fill them.
    Uses morphological closing to merge nearby letter components (e.g.,
    "(", "R", ")") into single blobs before size-filtering.
    """
    import cv2
    import numpy as np

    if image.mode != 'RGB':
        image = image.convert('RGB')

    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    # Threshold to find ink
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Close small gaps to merge "(", "R", ")" into single blobs.
    # A 5x3 kernel bridges horizontal gaps typical of italic text spacing.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Find connected components on closed image
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        closed, connectivity=8
    )

    result = img.copy()
    mask = np.zeros((h, w), dtype=np.uint8)

    # Stereodescriptor label: "(R)" or "(S)" is typically 15-45px wide, 8-20px tall
    min_cw, max_cw = 8, int(w * 0.14)
    min_ch, max_ch = 6, int(h * 0.10)

    for i in range(1, num_labels):  # skip background
        cx = stats[i, cv2.CC_STAT_LEFT]
        cy = stats[i, cv2.CC_STAT_TOP]
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if not (min_cw <= cw <= max_cw and min_ch <= ch <= max_ch):
            continue

        # Aspect ratio: wider than tall (text-like), but not extremely elongated
        aspect = cw / max(ch, 1)
        if not (0.6 <= aspect <= 3.5):
            continue

        # Fill ratio of the closed blob
        fill = area / max(cw * ch, 1)
        if not (0.20 <= fill <= 0.85):
            continue

        # Check the original (non-closed) binary for the actual ink in this region.
        # Stereodescriptors have multiple sub-components: "(", letter, ")"
        roi = binary[cy:cy+ch, cx:cx+cw]
        roi_nlabels = cv2.connectedComponents(roi, connectivity=8)[0]
        # "(R)" typically has 3-5 original components: "(", "R", ")", and any
        # serifs. Single solid blobs (atoms, bonds) have 1-2.
        if roi_nlabels < 3:
            continue

        # Ink density in original ROI (stereodescriptors are moderately dense text)
        orig_fill = cv2.countNonZero(roi) / max(cw * ch, 1)
        if orig_fill < 0.08 or orig_fill > 0.60:
            continue

        # Mark for masking
        pad = 2
        y0 = max(cy - pad, 0)
        y1 = min(cy + ch + pad, h)
        x0 = max(cx - pad, 0)
        x1 = min(cx + cw + pad, w)
        mask[y0:y1, x0:x1] = 255

    # Safety: don't mask if we'd remove too much ink (>15%)
    total_ink = cv2.countNonZero(binary)
    masked_ink = cv2.countNonZero(binary & mask)
    if total_ink > 0 and masked_ink / total_ink > 0.15:
        return image

    # Apply mask
    result[mask > 0] = 255
    return Image.fromarray(result)


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
            if ckpt_path == "patent_sft_final.pth":
                url = "https://huggingface.co/hanmozhang1984/MolSight-patent-finetuned/resolve/main/patent_sft_final.pth?download=true"
                print(f"Downloading fine-tuned MolSight weights from HuggingFace...", file=sys.stderr, flush=True)
            else:
                url = "https://huggingface.co/Robert-zwr/MolSight/resolve/main/pubchem_uspto_smiles_edges_30.pth?download=true"
                print(f"WARNING: Fine-tuned checkpoint not found, downloading base MolSight weights...", file=sys.stderr, flush=True)
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
                avg_logprob = batch_preds["avg_logprob"][0]
                smiles, _, success = _postprocess_smiles(raw_smiles)

                if Chem is not None:
                    try:
                        smiles = Chem.CanonSmiles(smiles)
                    except Exception:
                        pass

                if smiles and len(smiles.strip()) > 0:
                    print(f"{{smiles.strip()}}\\t{{avg_logprob:.6f}}", flush=True)
                else:
                    print("NONE", flush=True)
            except Exception as e:
                print(f"NONE", flush=True)
                print(f"Error: {{e}}", file=sys.stderr, flush=True)
    """)

    # Default WORKER_SCRIPT for backward compatibility (used by SubprocessPredictor)
    WORKER_SCRIPT = _WORKER_SCRIPT_TEMPLATE.format(
        molsight_dir=MOLSIGHT_DIR,
        checkpoint_path="patent_sft_final.pth",
        use_lora="False",
    )

    def __init__(
        self,
        checkpoint_path: str = "patent_sft_final.pth",
        confidence_threshold: float = -0.012,
        hybrid_fallback: bool = False,
        vision_model: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
    ):
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
        self._claude_fallback = None
        self._confidence_threshold = confidence_threshold
        self._hybrid_fallback = hybrid_fallback
        self._vision_model = vision_model
        self._api_key = api_key
        # Check if text masking is available
        try:
            from pdf_to_smiles.core.text_masker import is_available as _text_masker_available
            self._mask_text = _text_masker_available()
        except Exception:
            self._mask_text = False

    def predict_single_with_confidence(self, image: Image.Image) -> Tuple[Optional[str], float]:
        """Predict SMILES and return (smiles, avg_logprob) confidence score.

        The worker subprocess outputs 'SMILES\\tavg_logprob'. This method
        parses both parts. Returns (None, -inf) on failure.
        """
        import tempfile
        fd, path = tempfile.mkstemp(suffix=".png", prefix="pred_")
        os.close(fd)
        try:
            image.save(path)
            with self._lock:
                self._ensure_running()
                self._process.stdin.write(path + "\n")
                self._process.stdin.flush()
                result = self._readline_with_timeout(120)
            if result is None or result == "NONE" or not result:
                return None, float('-inf')
            # Parse SMILES\tconfidence format
            if '\t' in result:
                parts = result.split('\t', 1)
                smiles = parts[0]
                try:
                    confidence = float(parts[1])
                except (ValueError, IndexError):
                    confidence = float('-inf')
                return smiles if smiles else None, confidence
            # Fallback: no tab means old format (no confidence)
            return result, float('-inf')
        except (BrokenPipeError, OSError) as e:
            logger.warning("Subprocess crashed, will restart: %s", e)
            self._process = None
            return None, float('-inf')
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def predict_single(self, image: Image.Image) -> Optional[str]:
        """Predict SMILES from image (discards confidence)."""
        smiles, _ = self.predict_single_with_confidence(image)
        return smiles

    def _get_molscribe_fallback(self):
        """Lazy-load MolScribe as fallback predictor."""
        if self._molscribe_fallback is None:
            from .lightweight_predictor import LightweightPredictor
            self._molscribe_fallback = LightweightPredictor()
        return self._molscribe_fallback

    def _get_claude_fallback(self):
        """Lazy-load Claude Vision API predictor."""
        if self._claude_fallback is None:
            from .claude_vision_predictor import ClaudeVisionPredictor
            self._claude_fallback = ClaudeVisionPredictor(
                api_key=self._api_key,
                model=self._vision_model,
            )
        return self._claude_fallback

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
        if best_connected:
            return best_connected
        if best_disconnected:
            return _keep_largest_fragment(best_disconnected)
        return None

    @staticmethod
    def _postprocess(smiles: Optional[str]) -> Optional[str]:
        """Apply all SMILES post-processing fixes."""
        if not smiles:
            return smiles
        smiles = _fix_sulfur_valence(smiles)
        smiles = _fix_deuterium_wildcards(smiles)
        return smiles

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
        5. If hybrid_fallback enabled and best confidence < threshold → Claude Vision API

        At each step, apply SCF3 + deuterium post-processing and prefer
        connected SMILES.
        """
        smiles, confidence = self.predict_with_confidence(image, high_accuracy=high_accuracy)

        # Confidence-based API fallback
        if (
            self._hybrid_fallback
            and confidence < self._confidence_threshold
            and smiles is not None
        ):
            try:
                api = self._get_claude_fallback()
                api_result = api.predict(image)
                if api_result and _is_valid_smiles(api_result):
                    logger.info(
                        "Claude Vision fallback used (confidence=%.3f < %.3f): %s → %s",
                        confidence, self._confidence_threshold,
                        (smiles or "NONE")[:40], api_result[:40],
                    )
                    return api_result
            except Exception as e:
                logger.warning("Claude Vision fallback failed: %s", e)

        return smiles

    def predict_with_confidence(
        self,
        image: Image.Image,
        high_accuracy: bool = False,
    ) -> Tuple[Optional[str], float]:
        """Predict SMILES with full pipeline, returning (smiles, best_confidence).

        Same multi-variant pipeline as predict(), but returns the confidence
        score of the best result for downstream routing decisions.
        """
        molsight_results: list[Tuple[Optional[str], float]] = []

        # Step 0: Tighten oversized crops before any prediction attempt.
        from pdf_to_smiles.core.image_cleaner import autocrop_structure
        image = autocrop_structure(image)

        # Step 1: Raw image — MolSight's internal transforms handle it best
        raw_smiles, raw_conf = self.predict_single_with_confidence(image)
        result = self._postprocess(raw_smiles)
        molsight_results.append((result, raw_conf))

        # Early return if we got a valid connected SMILES
        if result and _is_valid_smiles(result) and '.' not in result:
            return result, raw_conf

        # Step 2: Cleaned image (remove bleeding lines, mask text)
        cleaned = self._preprocess(image)
        raw_smiles, raw_conf = self.predict_single_with_confidence(cleaned)
        result = self._postprocess(raw_smiles)
        molsight_results.append((result, raw_conf))

        if result and _is_valid_smiles(result) and '.' not in result:
            return result, raw_conf

        # Step 3: CLAHE enhancement on cleaned image
        enhanced = self._enhance_image(cleaned, aggressive=False)
        raw_smiles, raw_conf = self.predict_single_with_confidence(enhanced)
        result = self._postprocess(raw_smiles)
        molsight_results.append((result, raw_conf))

        # Check if any MolSight variant gave a usable result — pick highest confidence
        best_smiles = None
        best_conf = float('-inf')
        for smi, conf in molsight_results:
            if _is_valid_smiles(smi) and conf > best_conf:
                best_smiles = smi
                best_conf = conf

        # Also consider connected vs disconnected preference
        best_from_results = self._best_result(*(s for s, _ in molsight_results))
        if best_from_results is not None:
            # Use confidence from the matching result
            for smi, conf in molsight_results:
                if smi == best_from_results or self._postprocess(smi) == best_from_results:
                    best_conf = max(best_conf, conf)
            return best_from_results, best_conf

        # Step 4: MolScribe fallback — only if ALL MolSight variants failed
        try:
            fallback = self._get_molscribe_fallback()
            result = fallback.predict(image, high_accuracy=high_accuracy)
            if result:
                logger.info("MolScribe fallback succeeded where MolSight failed")
                # MolScribe doesn't provide confidence — use -inf to signal unknown
                return result, float('-inf')
        except Exception as e:
            logger.warning("MolScribe fallback failed: %s", e)

        return None, float('-inf')

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
        self._claude_fallback = None
