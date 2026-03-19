"""VLM-based final product extraction from synthesis scheme crops.

When a detected crop covers a large portion of a page (likely a multi-step
synthesis scheme rather than a single structure), this module asks a VLM to
identify and return the SMILES of just the final product.

Backend priority:
  1. MLX-VLM (local, free, fast on Apple Silicon)
  2. Claude Haiku (cloud fallback if MLX-VLM unavailable)

This avoids wasting 3-4 MolSight retries on impossible scheme crops.
"""

from __future__ import annotations

import base64
import io
import logging
import re
from typing import Optional, Tuple

import requests
from PIL import Image

logger = logging.getLogger(__name__)

# Crop must cover at least this fraction of the page to be considered a scheme
_MIN_WIDTH_FRAC = 0.55
_MIN_HEIGHT_FRAC = 0.25
# AND area must be at least this fraction of the page
_MIN_AREA_FRAC = 0.20

_MAX_IMAGE_DIM = 1024  # MLX-VLM optimal; also fine for Haiku

_SCHEME_EXTRACT_PROMPT = """\
This image shows a chemical synthesis scheme from a patent — multiple structures \
connected by reaction arrows, with reagent labels and intermediates.

Identify the FINAL PRODUCT of this synthesis scheme. The final product is \
typically:
- The LAST structure in the arrow sequence (no outgoing reaction arrow)
- Often labeled "Compound X", "Example X", or boxed/highlighted
- Usually at the bottom-right or end of the reaction flow

Return ONLY the SMILES string of the final product structure. \
Do NOT include intermediates, starting materials, or reagents. \
If you cannot confidently identify a single final product, return "NONE".

Respond with just the SMILES string on a single line, nothing else."""


def is_available() -> bool:
    """Check if any VLM backend is available (MLX-VLM or Claude Haiku)."""
    return _mlx_available() or _haiku_available()


def is_scheme_crop(
    crop_box: Optional[Tuple[int, int, int, int]],
    page_width: int,
    page_height: int,
) -> bool:
    """Check whether a crop is large enough to likely be a synthesis scheme.

    Args:
        crop_box: (x1, y1, x2, y2) bounding box in page-image pixels.
        page_width: Full page image width.
        page_height: Full page image height.

    Returns:
        True if the crop looks scheme-sized.
    """
    if crop_box is None:
        return False

    x1, y1, x2, y2 = crop_box
    crop_w = x2 - x1
    crop_h = y2 - y1

    width_frac = crop_w / max(page_width, 1)
    height_frac = crop_h / max(page_height, 1)
    area_frac = (crop_w * crop_h) / max(page_width * page_height, 1)

    return (
        width_frac >= _MIN_WIDTH_FRAC
        and height_frac >= _MIN_HEIGHT_FRAC
        and area_frac >= _MIN_AREA_FRAC
    )


def extract_final_product_smiles(crop_image: Image.Image) -> Optional[str]:
    """Extract the final product SMILES from a synthesis scheme crop.

    Tries MLX-VLM first (local, free), falls back to Claude Haiku.

    Args:
        crop_image: PIL Image of the synthesis scheme crop.

    Returns:
        SMILES string of the final product, or None if extraction fails.
    """
    resized = _resize_for_api(crop_image.convert("RGB"))
    image_b64 = _encode_image(resized)

    # Try MLX-VLM first
    if _mlx_available():
        result = _extract_via_mlx(image_b64)
        if result is not None:
            return result
        logger.debug("MLX-VLM scheme extraction returned nothing, trying Haiku fallback")

    # Fallback to Claude Haiku
    if _haiku_available():
        return _extract_via_haiku(image_b64)

    return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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
    return image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def _parse_smiles_response(raw: str) -> Optional[str]:
    """Parse a VLM response into a SMILES string, or None."""
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        # Skip markdown code fences
        if line.startswith("```"):
            continue
        if line.upper() == "NONE":
            return None
        # Basic SMILES sanity: should contain atoms, not prose
        if re.search(r'[A-Z][a-z]?\(', line) or re.search(r'[cnos]\d?', line):
            return line
        # Also accept if it looks SMILES-ish (brackets, digits, no spaces)
        if ' ' not in line and len(line) >= 5 and re.search(r'[CNOScnos]', line):
            return line
        break
    return None


# ---------------------------------------------------------------------------
# MLX-VLM backend
# ---------------------------------------------------------------------------

def _get_mlx_settings() -> Tuple[str, str]:
    """Get MLX-VLM endpoint and model from InferenceSettings."""
    try:
        from .inference_settings import InferenceSettings
        settings = InferenceSettings.get_instance()
        return settings.mlx_endpoint, settings.mlx_model
    except Exception:
        return "http://localhost:8000", "mlx-community/Qwen3-VL-8B-Instruct-4bit"


def _mlx_available() -> bool:
    """Check if MLX-VLM server is reachable."""
    base_url, _ = _get_mlx_settings()
    try:
        resp = requests.get(f"{base_url}/health", timeout=3)
        if resp.status_code == 200:
            return True
        resp = requests.get(f"{base_url}/v1/models", timeout=3)
        return resp.status_code == 200
    except (requests.ConnectionError, requests.Timeout, Exception):
        return False


def _extract_via_mlx(image_b64: str) -> Optional[str]:
    """Call MLX-VLM to extract final product SMILES."""
    base_url, model = _get_mlx_settings()
    try:
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": _SCHEME_EXTRACT_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_b64}"
                                },
                            },
                        ],
                    }
                ],
                "max_tokens": 512,
                "temperature": 0,
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        raw = data["choices"][0]["message"]["content"].strip()
        logger.debug("MLX-VLM scheme extractor raw: %s", raw)
        return _parse_smiles_response(raw)
    except Exception as e:
        logger.warning("MLX-VLM scheme extraction failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Claude Haiku backend (fallback)
# ---------------------------------------------------------------------------

_HAIKU_MODEL = "claude-haiku-4-5-20251001"


def _haiku_available() -> bool:
    """Check if anthropic SDK is installed and ANTHROPIC_API_KEY is set."""
    try:
        import anthropic  # noqa: F401
        import os
        return bool(os.environ.get("ANTHROPIC_API_KEY"))
    except ImportError:
        return False


def _extract_via_haiku(image_b64: str) -> Optional[str]:
    """Call Claude Haiku to extract final product SMILES."""
    try:
        import anthropic

        client = anthropic.Anthropic()
        response = client.messages.create(
            model=_HAIKU_MODEL,
            max_tokens=512,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": _SCHEME_EXTRACT_PROMPT,
                        },
                    ],
                }
            ],
        )

        raw = response.content[0].text.strip()
        logger.debug("Haiku scheme extractor raw: %s", raw)
        return _parse_smiles_response(raw)
    except Exception as e:
        logger.warning("Haiku scheme extraction failed: %s", e)
        return None
