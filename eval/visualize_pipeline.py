#!/usr/bin/env python3
"""Full-pipeline visualization dashboard.

Runs structure detection → compound classification → SMILES prediction →
formula extraction & validation on selected patent pages, then generates
a self-contained HTML dashboard with annotated images and results.

Usage:
    python eval/visualize_pipeline.py                # full pipeline
    python eval/visualize_pipeline.py --no-formula   # skip formula extraction
"""

import argparse
import base64
import html
import io
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

# Add project src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pypdfium2 as pdfium
from PIL import Image, ImageDraw, ImageFont

from pdf_to_smiles.core.docling_classifier import DoclingClassifier
from pdf_to_smiles.core.docling_detector import DoclingDetector
from pdf_to_smiles.core.formula_extractor import (
    FormulaReference,
    _EXTRACTION_PROMPT,
    _parse_response,
    _resize_for_api,
)
from pdf_to_smiles.core.formula_validator import FormulaValidator, correct_compound_ids
from pdf_to_smiles.core.inference_provider import InferenceProvider
from pdf_to_smiles.core.mlx_compound_classifier import MLXVLMCompoundClassifier
from pdf_to_smiles.core.smiles_validator import SMILESValidator

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

PDF_DIR = os.path.expanduser("~/Downloads/Sample patents for testing")
DPI_SCALE = 200 / 72
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
OUTPUT_HTML = os.path.join(OUTPUT_DIR, "pipeline_dashboard.html")

# Pages to process per patent (page_num is 1-indexed)
PATENTS_TO_PROCESS = {
    "Merck": {
        "filename": "WO_2026024861_A1.pdf",
        "pages": [108, 120, 155],
    },
    "GLP": {
        "filename": "US20240366598A1.pdf",
        "pages": [98, 99, 130],
    },
}

# MLX-VLM config (local Qwen model — zero cost, no API key needed)
MLX_BASE_URL = "http://localhost:8000"
MLX_MODEL = "mlx-community/Qwen3-VL-8B-Instruct-4bit"
MLX_REQUEST_TIMEOUT = 300

# Colors for bounding boxes (one per structure, cycled)
BOX_COLORS = [
    (255, 80, 80),    # red
    (80, 180, 255),   # blue
    (80, 220, 80),    # green
    (255, 180, 40),   # orange
    (200, 100, 255),  # purple
    (255, 220, 40),   # yellow
    (40, 220, 200),   # teal
    (255, 120, 180),  # pink
]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def pil_to_base64(img: Image.Image, fmt: str = "PNG", quality: int = 85) -> str:
    """Encode a PIL image to a base64 data URI."""
    buf = io.BytesIO()
    if fmt == "JPEG":
        img = img.convert("RGB")
        img.save(buf, format=fmt, quality=quality)
    else:
        img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def thumbnail(img: Image.Image, max_dim: int = 200) -> Image.Image:
    """Return a resized copy that fits within max_dim x max_dim."""
    img = img.copy()
    img.thumbnail((max_dim, max_dim), Image.LANCZOS)
    return img


def annotate_page_image(
    page_image: Image.Image,
    boxes: List[Tuple[int, int, int, int]],
    labels: List[str],
) -> Image.Image:
    """Draw numbered colored bounding boxes on a copy of the page image."""
    annotated = page_image.copy().convert("RGBA")
    overlay = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    draw = ImageDraw.Draw(annotated)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except (OSError, IOError):
        font = ImageFont.load_default()

    for i, (box, label) in enumerate(zip(boxes, labels)):
        color = BOX_COLORS[i % len(BOX_COLORS)]
        x1, y1, x2, y2 = box
        fill_color = color + (40,)
        overlay_draw.rectangle(box, fill=fill_color)
        draw.rectangle(box, outline=color, width=3)
        # Label background
        text = f"[{i}] {label}"
        draw.rectangle([(x1, y1 - 24), (x1 + len(text) * 10, y1)], fill=color)
        draw.text((x1 + 4, y1 - 22), text, fill=(255, 255, 255), font=font)

    annotated = Image.alpha_composite(annotated, overlay).convert("RGB")
    return annotated


def validation_badge(status: str) -> str:
    """Return an HTML badge for a formula validation status."""
    styles = {
        "match": ("background:#22c55e;color:#fff", "MATCH"),
        "mass_only_match": ("background:#eab308;color:#fff", "MASS MATCH"),
        "mismatch": ("background:#ef4444;color:#fff", "MISMATCH"),
        "no_reference": ("background:#9ca3af;color:#fff", "NO REF"),
    }
    style, text = styles.get(status, ("background:#6b7280;color:#fff", status.upper()))
    return f'<span style="display:inline-block;padding:2px 8px;border-radius:4px;font-size:12px;font-weight:600;{style}">{text}</span>'


# ------------------------------------------------------------------
# MLX-based formula extractor (local, zero cost)
# ------------------------------------------------------------------


class MLXFormulaExtractor:
    """Extract formulas using the local MLX-VLM server (OpenAI-compatible API)."""

    def __init__(self, base_url: str = MLX_BASE_URL, model: str = MLX_MODEL):
        self._base_url = base_url.rstrip("/")
        self._model = model

    def extract_from_page(
        self, page_image: Image.Image, page_num: int
    ) -> List[FormulaReference]:
        """Send page image to local MLX-VLM with the formula extraction prompt."""
        import requests as req

        img = _resize_for_api(page_image)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        try:
            response = req.post(
                f"{self._base_url}/v1/chat/completions",
                json={
                    "model": self._model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": _EXTRACTION_PROMPT},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{image_b64}"
                                    },
                                },
                            ],
                        }
                    ],
                    "max_tokens": 2048,
                    "temperature": 0,
                },
                timeout=MLX_REQUEST_TIMEOUT,
            )
            response.raise_for_status()
        except Exception as e:
            print(f"    MLX formula extraction failed for page {page_num}: {e}")
            return []

        try:
            data = response.json()
            raw_text = data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            print(f"    MLX formula response parse error for page {page_num}: {e}")
            return []

        return _parse_response(raw_text, page_num)


# ------------------------------------------------------------------
# Pipeline per page
# ------------------------------------------------------------------


def process_page(
    page_image: Image.Image,
    page_num: int,
    detector: DoclingDetector,
    classifier: MLXVLMCompoundClassifier,
    inference: InferenceProvider,
    smiles_validator: SMILESValidator,
    formula_extractor: Optional[MLXFormulaExtractor],
    formula_validator: FormulaValidator,
    skip_formula: bool = False,
) -> dict:
    """Run the full pipeline on a single page. Returns a result dict."""
    result = {
        "page_num": page_num,
        "structures": [],
        "formula_refs": [],
        "annotated_image": None,
    }

    # 1. Detect structures
    print(f"    Detecting structures...")
    detections = detector.detect_structures_with_boxes(page_image, page_num)
    if not detections:
        print(f"    No structures found on page {page_num}")
        result["annotated_image"] = page_image
        return result

    crops = [d[0] for d in detections]
    boxes = [d[1] for d in detections]
    print(f"    Found {len(detections)} structures")

    # 2. Classify compounds
    print(f"    Classifying compounds...")
    try:
        classifications = classifier.classify_page_structures(
            page_image, boxes, page_num=page_num
        )
    except Exception as e:
        print(f"    Classification failed: {e}")
        classifications = [{"type": "other", "id": None}] * len(boxes)

    # 3. Extract formulas from page (unless skipped)
    formula_refs = []
    if not skip_formula and formula_extractor is not None:
        print(f"    Extracting formulas...")
        try:
            formula_refs = formula_extractor.extract_from_page(page_image, page_num)
        except Exception as e:
            print(f"    Formula extraction failed: {e}")
    elif skip_formula:
        print(f"    Skipping formula extraction (--no-formula)")
    result["formula_refs"] = formula_refs

    # 4. Per-structure: predict SMILES, compute properties (no formula validation yet)
    for i, (crop, box) in enumerate(zip(crops, boxes)):
        cls = classifications[i] if i < len(classifications) else {"type": "other", "id": None}
        compound_id = cls.get("id")
        compound_type = cls.get("type", "other")

        struct_result = {
            "index": i,
            "box": box,
            "compound_id": compound_id,
            "compound_type": compound_type,
            "smiles": None,
            "smiles_valid": False,
            "canonical_smiles": None,
            "rdkit_image": None,
            "crop_image": crop,
            "properties": {},
            "formula_validation": None,
        }

        # Predict SMILES
        label = compound_id or f"#{i}"
        print(f"    [{i}] Predicting SMILES for {label}...")
        try:
            smiles = inference.predict_smiles(crop)
        except Exception as e:
            print(f"    [{i}] Prediction failed: {e}")
            smiles = None
        struct_result["smiles"] = smiles

        if smiles:
            # Validate and render
            is_valid, canonical, rdkit_img = smiles_validator.validate_and_render(smiles)
            struct_result["smiles_valid"] = is_valid
            struct_result["canonical_smiles"] = canonical
            struct_result["rdkit_image"] = rdkit_img

            if is_valid and canonical:
                props = smiles_validator.get_all_properties(canonical)
                struct_result["properties"] = props

        result["structures"].append(struct_result)

    # 5. Correct compound IDs via mass matching
    if formula_refs and result["structures"]:
        id_corrections = correct_compound_ids(result["structures"], formula_refs)
        for s in result["structures"]:
            if s["index"] in id_corrections:
                old_id = s["compound_id"]
                s["compound_id"] = id_corrections[s["index"]]
                print(f"    [{s['index']}] ID corrected: {old_id} -> {s['compound_id']}")

    # 6. Formula validation with corrected IDs
    for s in result["structures"]:
        if s["smiles_valid"] and s["canonical_smiles"] and formula_refs:
            try:
                fv_result = formula_validator.validate(
                    s["canonical_smiles"], formula_refs, compound_id=s["compound_id"]
                )
                s["formula_validation"] = fv_result
            except Exception as e:
                print(f"    [{s['index']}] Formula validation failed: {e}")

    # 7. Annotate page image with corrected labels
    labels = [s["compound_id"] or f"#{s['index']}" for s in result["structures"]]
    result["annotated_image"] = annotate_page_image(page_image, boxes, labels)

    return result


# ------------------------------------------------------------------
# HTML generation
# ------------------------------------------------------------------


def generate_html(all_results: Dict[str, List[dict]]) -> str:
    """Generate the self-contained HTML dashboard."""
    patent_ids = list(all_results.keys())

    # Build patent tab content
    patent_tabs_html = ""
    patent_content_html = ""

    for idx, patent_id in enumerate(patent_ids):
        active = "active" if idx == 0 else ""
        patent_tabs_html += f'<button class="tab-btn {active}" onclick="switchPatent(\'{patent_id}\')" id="tab-{patent_id}">{patent_id}</button>\n'

        display = "block" if idx == 0 else "none"
        pages_html = ""
        for page_result in all_results[patent_id]:
            pages_html += generate_page_html(page_result, patent_id)

        patent_content_html += f'<div class="patent-content" id="content-{patent_id}" style="display:{display}">{pages_html}</div>\n'

    return HTML_TEMPLATE.replace("{{PATENT_TABS}}", patent_tabs_html).replace(
        "{{PATENT_CONTENT}}", patent_content_html
    )


def generate_page_html(page_result: dict, patent_id: str) -> str:
    """Generate HTML for a single page result."""
    page_num = page_result["page_num"]
    structures = page_result["structures"]
    formula_refs = page_result["formula_refs"]
    annotated = page_result["annotated_image"]

    # Annotated page image (scale down for display)
    page_thumb = annotated.copy()
    page_thumb.thumbnail((800, 1200), Image.LANCZOS)
    page_b64 = pil_to_base64(page_thumb, "JPEG", quality=80)

    # Structure cards
    cards_html = ""
    for s in structures:
        cards_html += generate_structure_card(s)

    if not structures:
        cards_html = '<div class="no-data">No structures detected on this page.</div>'

    # Formula references section
    refs_html = ""
    if formula_refs:
        refs_html = '<div class="formula-refs"><h4>Extracted Formula References</h4><table class="refs-table"><thead><tr><th>Compound</th><th>Formula</th><th>Expected [M+H]⁺</th><th>Source</th><th>Confidence</th></tr></thead><tbody>'
        for ref in formula_refs:
            refs_html += f"<tr><td>{html.escape(ref.compound_id or '—')}</td>"
            refs_html += f"<td><code>{html.escape(ref.molecular_formula)}</code></td>"
            refs_html += f"<td>{ref.expected_mh_mass or '—'}</td>"
            refs_html += f"<td>{html.escape(ref.source)}</td>"
            refs_html += f"<td>{ref.confidence:.0%}</td></tr>"
        refs_html += "</tbody></table></div>"

    return f"""
    <div class="page-section">
        <h3>Page {page_num} — {len(structures)} structure(s), {len(formula_refs)} formula ref(s)</h3>
        <div class="page-layout">
            <div class="page-image">
                <img src="data:image/jpeg;base64,{page_b64}" alt="Page {page_num}" />
            </div>
            <div class="structures-grid">
                {cards_html}
            </div>
        </div>
        {refs_html}
    </div>
    """


def generate_structure_card(s: dict) -> str:
    """Generate an HTML card for a single structure result."""
    # Crop thumbnail
    crop = thumbnail(s["crop_image"], 150)
    crop_b64 = pil_to_base64(crop, "PNG")

    # RDKit rendered image
    rdkit_html = ""
    if s["rdkit_image"]:
        rdkit_thumb = thumbnail(s["rdkit_image"], 150)
        rdkit_b64 = pil_to_base64(rdkit_thumb, "PNG")
        rdkit_html = f'<img src="data:image/png;base64,{rdkit_b64}" alt="RDKit render" />'
    else:
        rdkit_html = '<div class="no-render">No RDKit render</div>'

    # Compound info
    compound_id = s["compound_id"] or "—"
    compound_type = s["compound_type"]
    type_badge = ""
    if compound_type == "example_compound":
        type_badge = '<span class="type-badge example">Example</span>'
    else:
        type_badge = '<span class="type-badge other">Other</span>'

    # SMILES
    smiles = s["smiles"] or "—"
    smiles_display = smiles[:50] + "..." if len(smiles) > 50 else smiles
    smiles_class = "valid" if s["smiles_valid"] else "invalid"

    # Formula validation
    fv_html = ""
    fv = s.get("formula_validation")
    if fv:
        fv_html = f'<div class="formula-val">{validation_badge(fv.status)}'
        if fv.reference_formula or fv.computed_formula:
            ref_f = fv.reference_formula or "—"
            comp_f = fv.computed_formula or "—"
            fv_html += f'<div class="formula-detail">Ref: <code>{html.escape(ref_f)}</code><br/>Comp: <code>{html.escape(comp_f)}</code></div>'
        if fv.mass_error_ppm is not None:
            fv_html += f'<div class="mass-error">Mass error: {fv.mass_error_ppm:.1f} ppm</div>'
        fv_html += "</div>"

    # Properties
    props = s.get("properties", {})
    mw = props.get("molecular_weight")
    formula = props.get("molecular_formula")
    props_html = ""
    if mw:
        props_html += f"<div>MW: {mw:.2f}</div>"
    if formula:
        props_html += f"<div>Formula: <code>{html.escape(formula)}</code></div>"

    box = s["box"]
    box_str = f"{box[0]},{box[1]}→{box[2]},{box[3]}"

    return f"""
    <div class="structure-card">
        <div class="card-header">
            <span class="struct-index">[{s['index']}]</span>
            <span class="compound-id">{html.escape(compound_id)}</span>
            {type_badge}
        </div>
        <div class="card-images">
            <div class="img-col">
                <div class="img-label">Original</div>
                <img src="data:image/png;base64,{crop_b64}" alt="Crop" />
            </div>
            <div class="img-col">
                <div class="img-label">RDKit</div>
                {rdkit_html}
            </div>
        </div>
        <div class="smiles-row {smiles_class}" title="{html.escape(smiles)}">
            {html.escape(smiles_display)}
        </div>
        <div class="card-details">
            {props_html}
            {fv_html}
        </div>
        <div class="box-info">{box_str}</div>
    </div>
    """


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Pipeline Dashboard — pdf_to_smiles</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #e2e8f0; padding: 20px; }
h1 { text-align: center; margin-bottom: 4px; font-size: 24px; color: #f8fafc; }
.subtitle { text-align: center; color: #94a3b8; margin-bottom: 20px; font-size: 14px; }
.tab-bar { display: flex; gap: 8px; justify-content: center; margin-bottom: 24px; }
.tab-btn { background: #1e293b; border: 1px solid #334155; color: #94a3b8; padding: 8px 24px; border-radius: 8px; cursor: pointer; font-size: 15px; font-weight: 600; transition: all 0.15s; }
.tab-btn:hover { background: #334155; color: #e2e8f0; }
.tab-btn.active { background: #3b82f6; border-color: #3b82f6; color: #fff; }
.page-section { background: #1e293b; border-radius: 12px; padding: 20px; margin-bottom: 24px; border: 1px solid #334155; }
.page-section h3 { color: #f8fafc; margin-bottom: 16px; font-size: 18px; border-bottom: 1px solid #334155; padding-bottom: 8px; }
.page-layout { display: flex; gap: 20px; align-items: flex-start; }
.page-image { flex: 0 0 auto; max-width: 45%; }
.page-image img { width: 100%; border-radius: 8px; border: 1px solid #334155; }
.structures-grid { flex: 1; display: flex; flex-wrap: wrap; gap: 12px; align-content: flex-start; }
.structure-card { background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 12px; width: calc(50% - 6px); min-width: 280px; }
.card-header { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }
.struct-index { color: #64748b; font-weight: 700; font-size: 13px; }
.compound-id { font-weight: 700; color: #f8fafc; font-size: 15px; }
.type-badge { display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 11px; font-weight: 600; }
.type-badge.example { background: #22c55e30; color: #4ade80; }
.type-badge.other { background: #64748b30; color: #94a3b8; }
.card-images { display: flex; gap: 8px; margin-bottom: 8px; }
.img-col { flex: 1; text-align: center; }
.img-col img { max-width: 100%; max-height: 150px; border-radius: 4px; border: 1px solid #334155; background: #fff; }
.img-label { font-size: 11px; color: #64748b; margin-bottom: 4px; }
.no-render { height: 80px; display: flex; align-items: center; justify-content: center; color: #475569; font-size: 12px; background: #1e293b; border-radius: 4px; }
.smiles-row { font-family: "SF Mono", "Fira Code", monospace; font-size: 11px; padding: 6px 8px; border-radius: 4px; word-break: break-all; margin-bottom: 8px; cursor: help; }
.smiles-row.valid { background: #22c55e15; color: #86efac; border: 1px solid #22c55e30; }
.smiles-row.invalid { background: #ef444415; color: #fca5a5; border: 1px solid #ef444430; }
.card-details { font-size: 12px; color: #94a3b8; }
.card-details code { background: #334155; padding: 1px 4px; border-radius: 2px; font-size: 11px; }
.formula-val { margin-top: 6px; }
.formula-detail { margin-top: 4px; font-size: 11px; }
.mass-error { margin-top: 2px; font-size: 11px; color: #cbd5e1; }
.box-info { font-size: 10px; color: #475569; margin-top: 6px; }
.no-data { color: #64748b; padding: 40px; text-align: center; width: 100%; }
.formula-refs { margin-top: 16px; padding-top: 12px; border-top: 1px solid #334155; }
.formula-refs h4 { color: #94a3b8; margin-bottom: 8px; font-size: 14px; }
.refs-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.refs-table th { background: #0f172a; color: #94a3b8; padding: 6px 10px; text-align: left; font-weight: 600; }
.refs-table td { padding: 6px 10px; border-top: 1px solid #1e293b; }
.refs-table code { background: #334155; padding: 1px 4px; border-radius: 2px; font-size: 12px; }
.timestamp { text-align: center; color: #475569; font-size: 12px; margin-top: 20px; }
</style>
</head>
<body>
<h1>Pipeline Dashboard</h1>
<div class="subtitle">pdf_to_smiles — Full Pipeline Visualization</div>
<div class="tab-bar">
{{PATENT_TABS}}
</div>
{{PATENT_CONTENT}}
<div class="timestamp">Generated: <span id="ts"></span></div>
<script>
document.getElementById('ts').textContent = new Date().toLocaleString();
function switchPatent(id) {
    document.querySelectorAll('.patent-content').forEach(el => el.style.display = 'none');
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
    document.getElementById('content-' + id).style.display = 'block';
    document.getElementById('tab-' + id).classList.add('active');
}
</script>
</body>
</html>
"""


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Full pipeline visualization dashboard")
    parser.add_argument(
        "--no-formula", action="store_true",
        help="Skip formula extraction and validation (classification + SMILES only)",
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Choose output filename based on mode
    if args.no_formula:
        output_html = os.path.join(OUTPUT_DIR, "pipeline_dashboard_no_formula.html")
    else:
        output_html = OUTPUT_HTML

    # Initialize shared components (all local — no API keys needed)
    print("Initializing inference provider...")
    inference = InferenceProvider()
    smiles_validator = SMILESValidator()
    formula_validator = FormulaValidator()

    print("Using local MLX-VLM for classification" +
          ("" if args.no_formula else " + formula extraction") + " (zero cost)")
    classifier = MLXVLMCompoundClassifier(base_url=MLX_BASE_URL, model=MLX_MODEL)
    formula_extractor = None if args.no_formula else MLXFormulaExtractor(
        base_url=MLX_BASE_URL, model=MLX_MODEL
    )

    all_results: Dict[str, List[dict]] = {}

    for patent_id, cfg in PATENTS_TO_PROCESS.items():
        pdf_path = os.path.join(PDF_DIR, cfg["filename"])
        if not os.path.exists(pdf_path):
            print(f"SKIP: {pdf_path} not found")
            continue

        print(f"\n{'=' * 50}")
        print(f"Processing patent: {patent_id}")
        print(f"{'=' * 50}")

        # Initialize docling classifier + detector per patent
        print(f"  Initializing Docling classifier...")
        doc_classifier = DoclingClassifier()
        doc_classifier.classify_pdf(pdf_path)
        detector = DoclingDetector(doc_classifier)
        doc = pdfium.PdfDocument(pdf_path)

        patent_results = []
        for page_num in cfg["pages"]:
            if page_num > len(doc):
                print(f"  Page {page_num} exceeds document length ({len(doc)}), skipping")
                continue

            print(f"\n  --- Page {page_num} ---")
            t0 = time.time()

            # Render page
            page = doc[page_num - 1]
            bitmap = page.render(scale=DPI_SCALE)
            page_image = bitmap.to_pil().convert("RGB")

            # Run pipeline
            page_result = process_page(
                page_image,
                page_num,
                detector,
                classifier,
                inference,
                smiles_validator,
                formula_extractor,
                formula_validator,
                skip_formula=args.no_formula,
            )
            patent_results.append(page_result)

            elapsed = time.time() - t0
            n_structs = len(page_result["structures"])
            n_valid = sum(1 for s in page_result["structures"] if s["smiles_valid"])
            n_refs = len(page_result["formula_refs"])
            print(f"    Done in {elapsed:.1f}s — {n_structs} structures ({n_valid} valid SMILES), {n_refs} formula refs")

        all_results[patent_id] = patent_results

    # Generate HTML
    print(f"\nGenerating HTML dashboard...")
    html_content = generate_html(all_results)
    with open(output_html, "w") as f:
        f.write(html_content)
    print(f"Dashboard saved to {output_html}")
    print(f"Open with: open {output_html}")


if __name__ == "__main__":
    main()
