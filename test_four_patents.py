#!/usr/bin/env python3
"""Diagnostic: test extraction on 4 new patent PDFs."""

import sys
sys.path.insert(0, 'src')

from pdf_to_smiles.core.biological_data_extractor import BiologicalDataExtractor
from pdf_to_smiles.workers.processing_worker import ProcessingWorker
import pypdfium2 as pdfium
import pytesseract
from PIL import Image
from pdf_to_smiles.utils.paths import configure_tesseract
configure_tesseract()

pdfs = [
    ('/Users/hanmozhang/Downloads/US11492346_KAT6pages.pdf', 'KAT6'),
    ('/Users/hanmozhang/Downloads/US20240366598A1_GLPpages.pdf', 'GLP'),
    ('/Users/hanmozhang/Downloads/US10934279_GLP1pages.pdf', 'GLP1'),
    ('/Users/hanmozhang/Downloads/US20230041385A1_HER2pages.pdf', 'HER2'),
]

for pdf_path, label in pdfs:
    print(f"\n{'#'*80}")
    print(f"### {label}: {pdf_path}")
    print(f"{'#'*80}")

    # 1. Page info and OCR text
    doc = pdfium.PdfDocument(pdf_path)
    print(f"Pages: {len(doc)}")
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        bitmap = page.render(scale=200/72)
        img = bitmap.to_pil().convert('RGB')
        w, h = img.size
        print(f"\n--- Page {page_idx+1} ({w}x{h}) ---")
        text = pytesseract.image_to_string(img)
        # Show first 800 chars
        print(text[:800])
        if len(text) > 800:
            print(f"... ({len(text)} total chars)")

    # 2. Compound detection per page
    print(f"\n--- Compound number detection ---")
    worker = ProcessingWorker()
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        bitmap = page.render(scale=200/72)
        img = bitmap.to_pil().convert('RGB')
        results = worker._detect_example_numbers_from_page(img, [None]*4, 1)
        print(f"  Page {page_idx+1}: {results}")
    doc.close()

    # 3. Bio data extraction
    print(f"\n--- Bio data extraction ---")
    extractor = BiologicalDataExtractor()
    all_data = extractor.extract_from_pdf_path(pdf_path)
    print(f"Total compounds: {len(all_data)}")
    for cid, bio in sorted(all_data.items(), key=lambda x: x[0]):
        assays = dict(list(bio.other_assays.items())[:5])
        legacy = f"ic50={bio.ic50}" if bio.ic50 else ""
        print(f"  '{cid}': {assays} {legacy}")

    print(f"\n--- Debug info ---")
    print(extractor.get_debug_info())
