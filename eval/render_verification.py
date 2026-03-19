#!/usr/bin/env python3
"""Render candidate pages for visual verification of structure counts."""

import os
import pypdfium2 as pdfium

PDF_DIR = os.path.expanduser("~/Downloads/Sample patents for testing")
OUT_DIR = os.path.join(os.path.dirname(__file__), "output", "_verification")
DPI_SCALE = 200 / 72

PAGES_TO_RENDER = {
    "TREM2": {
        "filename": "US11608344.pdf",
        "pages": [105, 110, 300, 385, 395],
    },
    "HER2": {
        "filename": "US20230041385A1.pdf",
        "pages": [85, 86, 205, 255, 256],
    },
    "GLP1": {
        "filename": "US10934279.pdf",
        "pages": [31, 55, 85, 93, 95],
    },
    "D5D": {
        "filename": "HK40078922A.pdf",
        "pages": [104, 105, 155, 182],
    },
    "KIF18": {
        "filename": "WO2021026098A1_kif18pages.pdf",
        "pages": [1],
    },
}

os.makedirs(OUT_DIR, exist_ok=True)

for patent_id, cfg in PAGES_TO_RENDER.items():
    pdf_path = os.path.join(PDF_DIR, cfg["filename"])
    if not os.path.exists(pdf_path):
        print(f"SKIP {patent_id}: {pdf_path} not found")
        continue

    doc = pdfium.PdfDocument(pdf_path)
    for page_num in cfg["pages"]:
        if page_num > len(doc):
            print(f"SKIP {patent_id} p{page_num}: exceeds doc length ({len(doc)})")
            continue

        page = doc[page_num - 1]
        bitmap = page.render(scale=DPI_SCALE)
        img = bitmap.to_pil()
        out_path = os.path.join(OUT_DIR, f"{patent_id}_page_{page_num:03d}.png")
        img.save(out_path)
        print(f"Saved {out_path} ({img.width}x{img.height})")

print(f"\nAll pages rendered to {OUT_DIR}")
