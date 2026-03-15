"""MLX-VLM-backed compound classification using OpenAI-compatible API.

Drop-in alternative to OllamaCompoundClassifier that uses MLX-VLM's
OpenAI-compatible server (mlx_vlm.server) instead of Ollama. ~2x faster
inference on Apple Silicon. Same interface: page image + structure bounding
boxes -> list of {"type": ..., "id": ...}.

Requires: mlx_vlm.server running locally (e.g.,
  python -m mlx_vlm.server --model mlx-community/Qwen3-VL-8B-Instruct-4bit
)
"""

import base64
import io
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import requests
from PIL import Image, ImageDraw, ImageFont

from .ollama_compound_classifier import (
    _encode_image,
    _extract_json,
    _load_prompt_template,
    _resize_for_api,
)

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://localhost:8000"
_DEFAULT_MODEL = "mlx-community/Qwen3-VL-8B-Instruct-4bit"
_REQUEST_TIMEOUT = 300  # seconds


def is_available(base_url: str = _DEFAULT_BASE_URL) -> bool:
    """Check if the MLX-VLM server is running.

    The server loads models on-demand when a chat completion request is made,
    so we only need to verify the server is reachable.
    """
    try:
        # Try /health first (mlx-vlm specific)
        resp = requests.get(f"{base_url}/health", timeout=5)
        if resp.status_code == 200:
            return True
        # Fallback: try /v1/models (always returns 200 if server is up)
        resp = requests.get(f"{base_url}/v1/models", timeout=5)
        return resp.status_code == 200
    except (requests.ConnectionError, requests.Timeout, Exception):
        return False


def check_mlx_status(base_url: str = _DEFAULT_BASE_URL) -> str:
    """Return a human-readable status string for MLX-VLM availability.

    Returns one of:
      - "ready" if server is running (models load on-demand)
      - "not_running" if connection refused
    """
    try:
        # Try /health first
        resp = requests.get(f"{base_url}/health", timeout=5)
        if resp.status_code == 200:
            return "ready"
        # Fallback: /v1/models
        resp = requests.get(f"{base_url}/v1/models", timeout=5)
        if resp.status_code == 200:
            return "ready"
        return "not_running"
    except requests.ConnectionError:
        return "not_running"
    except Exception:
        return "not_running"


class MLXVLMCompoundClassifier:
    """Classify detected structures using MLX-VLM via OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str = _DEFAULT_BASE_URL,
        model: str = _DEFAULT_MODEL,
        prompt_path: Optional[str] = None,
    ):
        self._base_url = base_url.rstrip("/")
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

        Same interface as OllamaCompoundClassifier.classify_page_structures().
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
        image_b64 = _encode_image(annotated)

        # Build section context
        section_context = self._build_section_context(page_num, section_bounds)

        # Build prompt
        prompt = self._prompt_template.format(n=n, section_context=section_context)

        # Call MLX-VLM via OpenAI-compatible chat completions endpoint
        try:
            response = requests.post(
                f"{self._base_url}/v1/chat/completions",
                json={
                    "model": self._model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{image_b64}"
                                    },
                                },
                            ],
                        }
                    ],
                    "max_tokens": 1024,
                    "temperature": 0,
                },
                timeout=_REQUEST_TIMEOUT,
            )
            response.raise_for_status()
        except requests.Timeout:
            logger.warning("MLX-VLM request timed out after %ds", _REQUEST_TIMEOUT)
            return [{"type": "other", "id": None}] * n
        except requests.RequestException as e:
            logger.warning("MLX-VLM request failed: %s", e)
            return [{"type": "other", "id": None}] * n

        # Parse response (standard OpenAI chat completions format)
        try:
            data = response.json()
            raw_text = data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            logger.warning("MLX-VLM response parse error: %s", e)
            return [{"type": "other", "id": None}] * n

        logger.debug("Page %s raw MLX-VLM response: %s", page_num or "?", raw_text)

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
            batch: List of dicts with keys: page_image, structure_boxes,
                page_num (optional), section_bounds (optional).
            max_parallel: Max concurrent requests.

        Returns:
            List of classification results (same order as input batch).
        """
        if not batch:
            return []

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
                    logger.warning(
                        "MLX batch classification failed for item %d: %s", idx, e
                    )

        return results

    @staticmethod
    def _build_section_context(
        page_num: Optional[int],
        section_bounds,
    ) -> str:
        """Build context string describing which patent section this page is in."""
        if section_bounds is None or not section_bounds.is_valid:
            if page_num is not None:
                return f"This is page {page_num} of the patent.\n"
            return ""

        parts = [f"This is page {page_num} of {section_bounds.total_pages}."]

        examples_start = section_bounds.examples_start
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
                    f"should be classified as \"other\"."
                )
            else:
                parts.append(
                    f"This page is in the EXAMPLES section (pages "
                    f"{examples_start}-{section_bounds.examples_end}). Structures "
                    f"here are likely example compounds, but still check for "
                    f"synthesis intermediates and Markush/generic structures."
                )

        return " ".join(parts) + "\n"

    @staticmethod
    def _parse_entry(value) -> Optional[Dict]:
        """Parse a single response entry into {"type": str, "id": Optional[str]}."""
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
        """Parse MLX-VLM's response into a list of classification dicts."""
        json_text = _extract_json(raw_text)

        try:
            data = json.loads(json_text)
        except json.JSONDecodeError:
            fixed = json_text.replace("'", '"')
            fixed = re.sub(r",\s*}", "}", fixed)
            fixed = re.sub(r",\s*]", "]", fixed)
            try:
                data = json.loads(fixed)
            except json.JSONDecodeError:
                logger.warning(
                    "Failed to parse MLX-VLM classifier response: %s", raw_text
                )
                return [{"type": "other", "id": None}] * n

        results = []
        for i in range(1, n + 1):
            value = data.get(str(i)) or data.get(i)
            entry = self._parse_entry(value)
            results.append(entry)

        # Fallback: use values in sorted key order
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

        results = [
            r if r is not None else {"type": "other", "id": None} for r in results
        ]

        return results
