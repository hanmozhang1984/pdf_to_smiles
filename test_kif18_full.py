#!/usr/bin/env python3
"""Diagnostic: full extraction pipeline on kif18pages PDF."""

import sys
sys.path.insert(0, 'src')

from pdf_to_smiles.core.biological_data_extractor import BiologicalDataExtractor
from pdf_to_smiles.workers.processing_worker import ProcessingWorker
import pypdfium2 as pdfium
from pdf_to_smiles.utils.paths import configure_tesseract
configure_tesseract()

pdf_path = '/Users/hanmozhang/Downloads/WO2021026098A1_kif18pages.pdf'

# Test bio data extraction
print("=== Bio data extraction ===")
extractor = BiologicalDataExtractor()
all_data = extractor.extract_from_pdf_path(pdf_path)
print(f"Total compounds: {len(all_data)}")
for cid, bio in sorted(all_data.items(), key=lambda x: x[0]):
    assays = dict(list(bio.other_assays.items())[:3])
    legacy = f"ic50={bio.ic50}" if bio.ic50 else ""
    print(f"  '{cid}': {assays} {legacy}")

print(f"\n=== Debug info ===")
print(extractor.get_debug_info())

# Test compound detection on each page
print(f"\n=== Compound number detection ===")
doc = pdfium.PdfDocument(pdf_path)
worker = ProcessingWorker()
for page_idx in range(len(doc)):
    page = doc[page_idx]
    bitmap = page.render(scale=200/72)
    img = bitmap.to_pil().convert('RGB')
    # Use 4 fake structures
    results = worker._detect_example_numbers_from_page(img, [None]*4, 1)
    print(f"  Page {page_idx + 1}: {results}")
doc.close()
