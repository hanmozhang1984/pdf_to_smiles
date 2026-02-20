# Contributing to PDF-to-SMILES

## Branching Strategy

All development happens on feature branches forked from the **`v1.1.0-biodata-fix`** baseline tag (commit `09e1897`). This tag is the common rollback point — if anything goes wrong on a branch, you can always reset to it.

```
master (stable)
  └── v1.1.0-biodata-fix (baseline tag)
        ├── feature/llm-compound-classification   (P1)
        ├── feature/text-cleanup                  (P2)
        ├── feature/llm-page-layout               (P3)
        ├── feature/paddleocr-text-masking         (P4)
        └── feature/layoutlm-finetuning            (P5)
```

### Feature Branches

| Branch | Priority | Owner | Description |
|--------|----------|-------|-------------|
| `feature/llm-compound-classification` | P1 | TBD | Add Claude Haiku classifier to label each detected structure as example compound, Markush, intermediate, or reference |
| `feature/text-cleanup` | P2 | TBD | Connected component analysis to remove overlaid text from structure images before OCSR |
| `feature/llm-page-layout` | P3 | TBD | Claude Vision for zero-shot page region classification (prototype) |
| `feature/paddleocr-text-masking` | P4 | TBD | PaddleOCR text detection + selective inpainting for structure images |
| `feature/layoutlm-finetuning` | P5 | TBD | LayoutLMv3 fine-tuning pipeline for production page layout analysis |

### File Ownership (to minimize merge conflicts)

Each priority primarily touches different files:

| Priority | Key files to modify | New files to create |
|----------|-------------------|-------------------|
| P1 | `workers/processing_worker.py` | `core/llm_classifier.py` |
| P2 | `core/smiles_predictor.py`, `core/lightweight_predictor.py` | `core/image_cleaner.py` |
| P3 | `core/page_classifier.py` | `core/llm_layout_analyzer.py` |
| P4 | `core/smiles_predictor.py` | `core/text_masker.py` |
| P5 | — | `core/layout_model.py`, `training/` directory |

> **Note:** P2 and P4 both touch `smiles_predictor.py`. Coordinate changes to this file to avoid merge conflicts. Ideally P2 merges first, then P4 rebases on top.

## Development Setup

### Prerequisites

- Python 3.9+
- Tesseract OCR (see README for platform-specific install)

### Setup

```bash
git clone https://github.com/hanmozhang1984/pdf_to_smiles.git
cd pdf_to_smiles

# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or: venv\Scripts\activate  # Windows

# Install in editable mode with dev dependencies
pip install -e .
pip install -r requirements-dev.txt
```

### Switching to your feature branch

```bash
git checkout feature/<your-branch-name>
```

## Running the App

```bash
# Launch the GUI
pdf-to-smiles

# Or run as module
python -m pdf_to_smiles
```

### Testing changes

Place test patent PDFs in a local directory and process them through the GUI. Compare output (SMILES strings, biological data) against known correct results.

```bash
# Run existing test scripts
python test_all_six.py
```

## PR Workflow

1. **Work on your feature branch** — commit early and often
2. **Keep up to date** — periodically rebase on `master` if it has moved forward:
   ```bash
   git fetch origin
   git rebase origin/master
   ```
3. **Open a Pull Request** to `master` when your feature is ready
4. **Get a review** from at least one other collaborator
5. **Squash-merge** into `master` to keep history clean

### Resetting to baseline

If you need to start over on your branch:

```bash
git checkout feature/<your-branch>
git reset --hard v1.1.0-biodata-fix
```

## Architecture Overview

```
src/pdf_to_smiles/
├── main.py                     # App entry point
├── gui/
│   ├── main_window.py          # PySide6 main window
│   └── inference_settings_dialog.py
├── core/
│   ├── pdf_processor.py        # PDF → page images
│   ├── page_classifier.py      # Classifies pages (structure vs text vs table)
│   ├── structure_detector.py   # Detects structure bounding boxes on a page
│   ├── compound_detector.py    # Detects compound numbers/labels
│   ├── smiles_predictor.py     # DECIMER/cloud OCSR (image → SMILES)
│   ├── lightweight_predictor.py # MolScribe local inference
│   ├── lightweight_detector.py # Local structure detection
│   ├── smiles_validator.py     # RDKit SMILES validation
│   ├── biological_data_extractor.py # Extract bioactivity tables
│   ├── export_handler.py       # CSV/Excel export
│   ├── inference_provider.py   # Cloud vs local inference routing
│   └── inference_settings.py   # Model/inference config
├── workers/
│   └── processing_worker.py    # Background processing thread (orchestrates pipeline)
├── cloud/
│   ├── client.py               # Cloud inference client
│   └── modal_app.py            # Modal serverless deployment
├── models/
│   └── extraction_result.py    # Data classes for results
└── utils/
    └── paths.py                # Path utilities
```

### Processing Pipeline

```
PDF → pdf_processor (page images)
    → page_classifier (which pages have structures?)
    → structure_detector (bounding boxes)
    → smiles_predictor (image → SMILES)
    → smiles_validator (RDKit check)
    → biological_data_extractor (tables)
    → export_handler (CSV/Excel)
```

All orchestrated by `processing_worker.py` in a background thread.
