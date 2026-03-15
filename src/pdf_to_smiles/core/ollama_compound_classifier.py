"""Ollama-backed compound classification using Qwen2.5-VL (local VLM).

Drop-in alternative to LLMCompoundClassifier that uses a free, local vision
language model via Ollama instead of Claude Haiku Vision.  Same interface:
page image + structure bounding boxes → list of {"type": ..., "id": ...}.

Supports loading an optimized prompt from file (e.g. produced by DSPy).
"""

import base64
import io
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import requests
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "qwen3.5:9b"
_OLLAMA_BASE_URL = "http://localhost:11434"
_REQUEST_TIMEOUT = 300  # seconds

# Reuse the same classification prompt as Claude path (can be overridden)
_DEFAULT_CLASSIFY_PROMPT = """\
You are classifying chemical structures on a pharmaceutical patent page.
I've drawn {n} red boxes around structures. Each box has a small red number label
(1, 2, 3, ...) at its top-left corner. Use THESE red box numbers as keys — NOT the
example/compound numbers printed on the page.

{section_context}
For each red-boxed structure, provide TWO things:
  (a) Its classification type (see rules below).
  (b) The compound/example number printed on the page near that structure — this is
      the patent's own number, NOT the red box number. Look for labels like
      "Example 25", "Ex. 7", "Compound 100-7", "Cpd. 12a", or a row number in a
      table. Return just the number/ID part (e.g., "25", "100-7", "12a").
      Return null if no compound number is visible near the structure (intermediates,
      Markush, reagents, etc. typically have no compound number).

Classify each structure by checking these rules IN ORDER:

1. INSIDE A TABLE with column headers like "Example", "Structure", "Name", "No."?
   -> "example_compound". Tables with example/compound numbers and structure
   drawings are ALWAYS example compounds, even if the page also has synthesis
   schemes elsewhere. Look for numbered rows (1, 2, 3... or 1a, 1b...).

2. Has an associated "Example X", "Ex. #", "Compound X", or "Cpd. X" label nearby?
   -> "example_compound" (the label may be separated by IUPAC names, molecular
   formulas, or synthesis text — look within ~2cm above or below the structure).

3. Is the FINAL product at the END of a multi-step synthesis scheme (last structure
   in a chain of arrows), AND is this page in the Examples section of the patent?
   -> "example_compound". The final product of an example synthesis IS the example
   compound. But intermediates in that same scheme are NOT — only the last product.

4. Everything else -> "other":
   - Markush/generic structures (R-groups like R\u00b9, R\u00b2, "Formula (I)", variable
     bonds shown as dashed lines, "wherein R is...")
   - Synthesis INTERMEDIATES — structures that appear BEFORE reaction arrows, or
     labeled with codes like "Int-X", "Intermediate", step numbers (Step 1, Step 2),
     or letter-number codes (A-1, G-2, C43, P1, SM-1)
   - Starting materials and reagents
   - Reference compounds or known drugs shown for comparison
   - Fragments with wavy bonds (partial structures)
   - Structures in patent Claims section (even if fully defined with no R-groups)
   - Structures labeled as "Formula (I)", "Formula (II)" etc. (general formulas)

IMPORTANT: A single page can have BOTH example compounds (in a table at the top)
AND synthesis intermediates (in a scheme below). Classify each structure individually.

Respond ONLY with JSON: {{"1": {{"type": "example_compound", "id": "25"}}, "2": {{"type": "other", "id": null}}, ...}}"""

# Max image dimension for Ollama (smaller than Claude's 1568 to speed up
# local inference — Qwen processes images proportionally to pixel count).
# Images above this threshold get downscaled; smaller images are untouched.
_MAX_IMAGE_DIM = 1024

# Default path for optimized prompt
_DEFAULT_PROMPT_PATH = os.path.join(
    os.path.expanduser("~"), ".pdf_to_smiles", "optimized_classifier_prompt.txt"
)


def is_available(model: str = _DEFAULT_MODEL) -> bool:
    """Check if Ollama is running and the model is downloaded.

    Returns True only if the Ollama server responds and the specified model
    is present in its model list.
    """
    try:
        resp = requests.get(f"{_OLLAMA_BASE_URL}/api/tags", timeout=5)
        if resp.status_code != 200:
            return False
        data = resp.json()
        model_names = [m.get("name", "") for m in data.get("models", [])]
        # Match with or without tag suffix
        return any(
            name == model or name.startswith(model + ":")
            or model.startswith(name.split(":")[0])
            for name in model_names
        )
    except (requests.ConnectionError, requests.Timeout, Exception):
        return False


def check_ollama_status(model: str = _DEFAULT_MODEL) -> str:
    """Return a human-readable status string for Ollama availability.

    Returns one of:
      - "ready" if Ollama is running and model is downloaded
      - "not_installed" if Ollama server is unreachable
      - "not_running" if connection refused (installed but not started)
      - "model_not_found" if server is up but model isn't downloaded
    """
    try:
        resp = requests.get(f"{_OLLAMA_BASE_URL}/api/tags", timeout=5)
        if resp.status_code != 200:
            return "not_running"
        data = resp.json()
        model_names = [m.get("name", "") for m in data.get("models", [])]
        if any(
            name == model or name.startswith(model + ":")
            or model.startswith(name.split(":")[0])
            for name in model_names
        ):
            return "ready"
        return "model_not_found"
    except requests.ConnectionError:
        return "not_running"
    except Exception:
        return "not_installed"


def _encode_image(pil_image: Image.Image) -> str:
    """Base64-encode a PIL image as PNG."""
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def _resize_for_api(image: Image.Image) -> Image.Image:
    """Resize image if longer edge exceeds the optimal limit."""
    w, h = image.size
    if max(w, h) <= _MAX_IMAGE_DIM:
        return image
    scale = _MAX_IMAGE_DIM / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return image.resize((new_w, new_h), Image.LANCZOS)


def _load_prompt_template(prompt_path: Optional[str] = None) -> str:
    """Load classification prompt from file, falling back to built-in default.

    Tries (in order):
      1. Explicit prompt_path argument
      2. Default path (~/.pdf_to_smiles/optimized_classifier_prompt.txt)
      3. Built-in _DEFAULT_CLASSIFY_PROMPT
    """
    paths_to_try = []
    if prompt_path:
        paths_to_try.append(prompt_path)
    paths_to_try.append(_DEFAULT_PROMPT_PATH)

    for path in paths_to_try:
        try:
            if os.path.isfile(path):
                with open(path, "r") as f:
                    content = f.read().strip()
                if content:
                    logger.info("Loaded optimized classifier prompt from %s", path)
                    return content
        except Exception:
            pass

    return _DEFAULT_CLASSIFY_PROMPT


def _extract_json(text: str) -> str:
    """Extract JSON from model output, handling markdown code blocks.

    Local models often wrap JSON in ```json ... ``` blocks.
    """
    # Try to extract from markdown code block first
    code_block_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if code_block_match:
        return code_block_match.group(1).strip()

    # Fall back to finding outermost { ... }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]

    return text


class OllamaCompoundClassifier:
    """Classify detected structures using a local VLM via Ollama."""

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        prompt_path: Optional[str] = None,
    ):
        self._model = model
        self._prompt_template = _load_prompt_template(prompt_path)

    def classify_page_structures(
        self,
        page_image: Image.Image,
        structure_boxes: List[Tuple[int, int, int, int]],
        page_num: Optional[int] = None,
        section_bounds=None,
    ) -> List[Dict]:
        """Classify each structure on a page and extract compound IDs.

        Same interface as LLMCompoundClassifier.classify_page_structures().
        """
        n = len(structure_boxes)
        if n == 0:
            return []

        # Draw numbered red boxes on a copy of the page image
        annotated = page_image.copy().convert("RGB")
        draw = ImageDraw.Draw(annotated)

        font = None
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        except (OSError, IOError):
            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20
                )
            except (OSError, IOError):
                font = ImageFont.load_default()

        for idx, (x1, y1, x2, y2) in enumerate(structure_boxes):
            label = str(idx + 1)
            for offset in range(3):
                draw.rectangle(
                    [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                    outline="red",
                )
            text_x = x1
            text_y = max(0, y1 - 24)
            bbox = draw.textbbox((text_x, text_y), label, font=font)
            draw.rectangle(
                [bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2],
                fill="white",
            )
            draw.text((text_x, text_y), label, fill="red", font=font)

        # Resize for API
        annotated = _resize_for_api(annotated)
        image_data = _encode_image(annotated)

        # Build section context
        section_context = self._build_section_context(page_num, section_bounds)

        # Build prompt
        prompt = self._prompt_template.format(n=n, section_context=section_context)

        # Call Ollama with streaming to avoid timeout during thinking phase.
        # Models like Qwen3.5 spend time "thinking" before emitting tokens;
        # a non-streaming request can time out waiting for the full response.
        try:
            response = requests.post(
                f"{_OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": self._model,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                            "images": [image_data],
                        }
                    ],
                    "stream": True,
                    "think": False,
                    "options": {
                        "temperature": 0,
                    },
                },
                timeout=_REQUEST_TIMEOUT,
                stream=True,
            )
            response.raise_for_status()
        except requests.Timeout:
            logger.warning("Ollama request timed out after %ds", _REQUEST_TIMEOUT)
            return [{"type": "other", "id": None}] * n
        except requests.RequestException as e:
            logger.warning("Ollama request failed: %s", e)
            return [{"type": "other", "id": None}] * n

        # Collect streamed response
        raw_text = ""
        try:
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    raw_text += chunk.get("message", {}).get("content", "")
                    if chunk.get("done"):
                        break
        except Exception as e:
            logger.warning("Error reading Ollama stream: %s", e)
            if not raw_text:
                return [{"type": "other", "id": None}] * n
        raw_text = raw_text.strip()
        logger.debug("Page %s raw Ollama response: %s", page_num or "?", raw_text)

        results = self._parse_response(raw_text, n)
        logger.debug(
            "Page %s classification: %s",
            page_num or "?",
            {str(i + 1): r for i, r in enumerate(results)},
        )
        return results

    def classify_batch(
        self,
        batch: List[dict],
        max_parallel: int = 4,
    ) -> List[Optional[List[Dict]]]:
        """Classify multiple pages in parallel for throughput.

        Args:
            batch: List of dicts, each with keys:
                - page_image: PIL Image
                - structure_boxes: List[Tuple[int,int,int,int]]
                - page_num: int (optional)
                - section_bounds: SectionBounds (optional)
            max_parallel: Max concurrent Ollama requests.

        Returns:
            List of classification results (same order as input batch).
            Each entry is a list of {"type": ..., "id": ...} or None on error.
        """
        if not batch:
            return []

        # For a single item, just call directly (no threading overhead)
        if len(batch) == 1:
            item = batch[0]
            result = self.classify_page_structures(
                item["page_image"],
                item["structure_boxes"],
                page_num=item.get("page_num"),
                section_bounds=item.get("section_bounds"),
            )
            return [result]

        results = [None] * len(batch)

        def _classify_one(idx: int, item: dict):
            return idx, self.classify_page_structures(
                item["page_image"],
                item["structure_boxes"],
                page_num=item.get("page_num"),
                section_bounds=item.get("section_bounds"),
            )

        with ThreadPoolExecutor(max_workers=max_parallel) as pool:
            futures = {
                pool.submit(_classify_one, i, item): i
                for i, item in enumerate(batch)
            }
            for future in as_completed(futures):
                try:
                    idx, result = future.result()
                    results[idx] = result
                except Exception as e:
                    idx = futures[future]
                    logger.warning("Batch classification failed for item %d: %s", idx, e)
                    # Leave as None

        return results

    @staticmethod
    def _build_section_context(
        page_num: Optional[int],
        section_bounds,
    ) -> str:
        """Build context string describing which patent section this page is in.

        Identical logic to LLMCompoundClassifier._build_section_context().
        """
        if section_bounds is None or not section_bounds.is_valid:
            if page_num is not None:
                return f"This is page {page_num} of the patent.\n"
            return ""

        parts = [f"This is page {page_num} of {section_bounds.total_pages}."]

        examples_start = section_bounds.examples_start
        examples_end = section_bounds.examples_end

        if page_num is not None and examples_start is not None:
            if page_num < examples_start:
                parts.append(
                    f"This page is BEFORE the Examples section (Examples start at "
                    f"page {examples_start}). Structures here are likely from the "
                    f"Description section — expect Markush/generic structures, "
                    f"general formulas, or illustrative schemes. Classify as \"other\" "
                    f"unless you see clear \"Example X\" labels."
                )
            elif (
                section_bounds.claims_start
                and page_num >= section_bounds.claims_start
            ):
                parts.append(
                    f"This page is in the CLAIMS section (Claims start at page "
                    f"{section_bounds.claims_start}). ALL structures on Claims pages "
                    f"should be classified as \"other\", even if they look like "
                    f"specific compounds."
                )
            else:
                parts.append(
                    f"This page is in the EXAMPLES section (pages "
                    f"{examples_start}-{examples_end}). Structures here are likely "
                    f"example compounds, but still check for synthesis intermediates "
                    f"and Markush/generic structures."
                )

        return " ".join(parts) + "\n"

    @staticmethod
    def _parse_entry(value) -> Optional[Dict]:
        """Parse a single response entry into {"type": str, "id": Optional[str]}.

        Handles both dict format and old flat string format.
        """
        if isinstance(value, dict):
            entry_type = value.get("type", "other")
            if entry_type not in ("example_compound", "other"):
                entry_type = "other"
            entry_id = value.get("id")
            if entry_id is not None:
                entry_id = str(entry_id).strip()
                if not entry_id or entry_id.lower() in ("null", "none"):
                    entry_id = None
            return {"type": entry_type, "id": entry_id}
        elif isinstance(value, str) and value in ("example_compound", "other"):
            return {"type": value, "id": None}
        return None

    def _parse_response(self, raw_text: str, n: int) -> List[Dict]:
        """Parse Ollama's response into a list of classification dicts.

        More robust than the Claude path: handles markdown code blocks,
        extra whitespace, and common local model formatting quirks.
        """
        json_text = _extract_json(raw_text)

        try:
            data = json.loads(json_text)
        except json.JSONDecodeError:
            # Try fixing common issues: single quotes, trailing commas
            fixed = json_text.replace("'", '"')
            fixed = re.sub(r",\s*}", "}", fixed)
            fixed = re.sub(r",\s*]", "]", fixed)
            try:
                data = json.loads(fixed)
            except json.JSONDecodeError:
                logger.warning(
                    "Failed to parse Ollama classifier response: %s", raw_text
                )
                return [{"type": "other", "id": None}] * n

        # Try direct box-number keys first ("1", "2", ...)
        results = []
        for i in range(1, n + 1):
            value = data.get(str(i)) or data.get(i)
            entry = self._parse_entry(value)
            results.append(entry)

        # Fallback: if keys don't match box numbers, use values in sorted key order
        if any(r is None for r in results) and len(data) == n:
            def _numeric_sort_key(kv):
                digits = re.sub(r"[^0-9]", "", str(kv[0]))
                return int(digits) if digits else float("inf")

            sorted_entries = []
            for _, v in sorted(data.items(), key=_numeric_sort_key):
                entry = self._parse_entry(v)
                sorted_entries.append(
                    entry if entry else {"type": "other", "id": None}
                )
            results = sorted_entries

        # Fill remaining Nones with default
        results = [
            r if r is not None else {"type": "other", "id": None} for r in results
        ]

        return results
