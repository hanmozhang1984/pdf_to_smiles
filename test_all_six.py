#!/usr/bin/env python3
"""Test extraction on all 6 patent PDFs."""

import sys
sys.path.insert(0, 'src')

from pdf_to_smiles.core.biological_data_extractor import BiologicalDataExtractor
from pdf_to_smiles.utils.paths import configure_tesseract
configure_tesseract()

pdfs = [
    ('/Users/hanmozhang/Downloads/US12291539B2_test2.pdf', 'TEST2'),
    ('/Users/hanmozhang/Downloads/WO2021026098A1_kif18pages.pdf', 'KIF18'),
    ('/Users/hanmozhang/Downloads/US11492346_KAT6pages.pdf', 'KAT6'),
    ('/Users/hanmozhang/Downloads/US20240366598A1_GLPpages.pdf', 'GLP'),
    ('/Users/hanmozhang/Downloads/US10934279_GLP1pages.pdf', 'GLP1'),
    ('/Users/hanmozhang/Downloads/US20230041385A1_HER2pages.pdf', 'HER2'),
]

for pdf_path, label in pdfs:
    print(f"\n{'='*60}")
    print(f"  {label}: {pdf_path.split('/')[-1]}")
    print(f"{'='*60}")

    extractor = BiologicalDataExtractor()
    all_data = extractor.extract_from_pdf_path(pdf_path)
    print(f"Total compounds: {len(all_data)}")
    for cid, bio in sorted(all_data.items(), key=lambda x: (len(x[0]), x[0])):
        assays = dict(list(bio.other_assays.items())[:4])
        legacy = f"ic50={bio.ic50}" if bio.ic50 else ""
        legacy += f"ec50={bio.ec50}" if bio.ec50 else ""
        print(f"  '{cid}': {assays} {legacy}")
