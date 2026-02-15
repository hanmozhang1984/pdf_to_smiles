#!/bin/bash
#
# Build script for PDF to SMILES (macOS/Linux)
#
# Prerequisites:
#   1. Python 3.9+ with pip
#   2. Tesseract OCR installed (brew install tesseract on macOS)
#   3. Virtual environment activated
#
# Usage:
#   ./build.sh          - Build the application
#   ./build.sh clean    - Clean build artifacts
#   ./build.sh install  - Install build dependencies

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
DIST_DIR="$PROJECT_DIR/dist"
BUILD_DIR="$PROJECT_DIR/build"

install_deps() {
    echo "Installing build dependencies..."
    pip install pyinstaller
    pip install -r requirements.txt
}

clean() {
    echo "Cleaning build artifacts..."
    rm -rf "$BUILD_DIR"
    rm -rf "$DIST_DIR"
    rm -f "$PROJECT_DIR"/*.spec.bak
    echo "Clean complete."
}

build() {
    echo ""
    echo "============================================"
    echo " PDF to SMILES - Build Script"
    echo "============================================"
    echo ""

    # Check for Tesseract
    if ! command -v tesseract &> /dev/null; then
        echo "WARNING: Tesseract not found in PATH."
        if [[ "$OSTYPE" == "darwin"* ]]; then
            echo "Install with: brew install tesseract"
        else
            echo "Install with: sudo apt-get install tesseract-ocr"
        fi
        echo ""
    fi

    # Check for virtual environment
    if [[ -z "$VIRTUAL_ENV" ]]; then
        echo "WARNING: No virtual environment detected."
        echo "It's recommended to build within the project's venv."
        echo ""
    fi

    echo "Building application with PyInstaller..."
    echo ""

    pyinstaller --clean pdf_to_smiles.spec

    echo ""
    echo "============================================"
    echo " Build complete!"
    echo "============================================"
    echo ""

    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "Output: $DIST_DIR/PDF-to-SMILES.app"
        echo ""
        echo "To run the application:"
        echo "  open $DIST_DIR/PDF-to-SMILES.app"
    else
        echo "Output directory: $DIST_DIR/PDF-to-SMILES"
        echo ""
        echo "To run the application:"
        echo "  $DIST_DIR/PDF-to-SMILES/PDF-to-SMILES"
    fi
    echo ""
}

case "$1" in
    clean)
        clean
        ;;
    install)
        install_deps
        ;;
    *)
        build
        ;;
esac
