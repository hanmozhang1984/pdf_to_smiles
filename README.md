# PDF to SMILES

A desktop GUI application that extracts chemical structure images from PDF files and converts them to SMILES strings using deep learning.

## Features

- Extract chemical structures from scientific PDFs
- Convert structure images to SMILES notation using DECIMER deep learning models
- Validate SMILES strings with RDKit
- Extract biological data and tables from documents
- Export results to CSV/Excel formats

## Requirements

- Python 3.9 or higher
- Tesseract OCR (see installation instructions below)

## Installation

### 1. Install Tesseract OCR

Tesseract OCR must be installed separately on your system:

**Windows:**
- Download the installer from [UB Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
- Run the installer and note the installation path (default: `C:\Program Files\Tesseract-OCR`)
- Add Tesseract to your system PATH, or set the `TESSDATA_PREFIX` environment variable

**macOS:**
```bash
brew install tesseract
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install tesseract-ocr
```

### 2. Install pdf-to-smiles

```bash
pip install pdf-to-smiles
```

Or install from source:

```bash
git clone https://github.com/chemcipher/pdf-to-smiles.git
cd pdf-to-smiles
pip install -e .
```

## Usage

Launch the application:

```bash
pdf-to-smiles
```

Or run as a module:

```bash
python -m pdf_to_smiles
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Third-Party Attributions

This software uses several open-source libraries. See the [NOTICE](NOTICE) file for full attribution details.

Key dependencies include:
- **DECIMER** - Deep learning models for chemical image recognition (MIT License)
- **RDKit** - Cheminformatics toolkit (BSD-3-Clause)
- **TensorFlow** - Machine learning framework (Apache 2.0)
- **PySide6** - Qt for Python GUI framework (LGPL v3)

If you use this software in academic work, please cite the DECIMER papers:

> Rajan, K., Zielesny, A. & Steinbeck, C. DECIMER 1.0: deep learning for chemical image recognition using transformers. *J Cheminform* 13, 61 (2021). https://doi.org/10.1186/s13321-021-00538-8

> Rajan, K., Brinkhaus, H.O., Sorokina, M. et al. DECIMER.ai: an open platform for automated optical chemical structure identification, segmentation and recognition in scientific publications. *Nat Commun* 14, 5045 (2023). https://doi.org/10.1038/s41467-023-40782-0

## Building Standalone Executables

To create a standalone executable that includes Tesseract OCR:

### Prerequisites

1. Install Tesseract OCR on your build machine
2. Install build dependencies:
   ```bash
   pip install pyinstaller
   ```

### Windows

```batch
build.bat
```

Output: `dist\PDF-to-SMILES\PDF-to-SMILES.exe`

### macOS

```bash
chmod +x build.sh
./build.sh
```

Output: `dist/PDF-to-SMILES.app`

### Linux

```bash
chmod +x build.sh
./build.sh
```

Output: `dist/PDF-to-SMILES/PDF-to-SMILES`

### Build Options

- `build.bat clean` / `./build.sh clean` - Remove build artifacts
- `build.bat install` / `./build.sh install` - Install build dependencies

The standalone executable bundles Tesseract OCR, so end users don't need to install it separately.
