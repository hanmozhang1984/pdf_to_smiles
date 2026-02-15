#!/usr/bin/env python3
"""Diagnostic: investigate kif18pages PDF compound/bio extraction."""

import sys
sys.path.insert(0, 'src')

import pypdfium2 as pdfium
from PIL import Image
import pytesseract
import re
from pdf_to_smiles.utils.paths import configure_tesseract
configure_tesseract()

pdf_path = '/Users/hanmozhang/Downloads/WO2021026098A1_kif18pages.pdf'

doc = pdfium.PdfDocument(pdf_path)
print(f"PDF: {pdf_path}")
print(f"Pages: {len(doc)}")

# Look at each page
for page_idx in range(len(doc)):
    page = doc[page_idx]
    bitmap = page.render(scale=200/72)
    img = bitmap.to_pil().convert('RGB')
    w, h = img.size
    print(f"\n{'='*60}")
    print(f"=== Page {page_idx + 1} ({w}x{h}) ===")

    # Full page OCR
    ocr = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

    # Show all words with positions
    print(f"\nAll OCR words:")
    for i in range(len(ocr['text'])):
        text = str(ocr['text'][i]).strip()
        conf = int(ocr['conf'][i]) if ocr['conf'][i] != '-1' else 0
        if text and conf > 15:
            x = ocr['left'][i]
            y = ocr['top'][i]
            x_pct = x / w * 100
            y_pct = y / h * 100
            print(f"  x={x:4d} ({x_pct:5.1f}%) y={y:4d} ({y_pct:5.1f}%) conf={conf:2d} '{text}'")

    # Also get plain text for context
    print(f"\nPlain text OCR:")
    text = pytesseract.image_to_string(img)
    print(text[:1000])
    print("...")

doc.close()
