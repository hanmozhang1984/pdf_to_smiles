#!/usr/bin/env bash
# Setup/rebuild MolSight environment.
#
# Usage:
#   bash setup_molsight.sh          # full setup (clone + venv + checkpoint)
#   bash setup_molsight.sh --venv   # rebuild venv only (if source is intact)
#
set -euo pipefail

MOLSIGHT_DIR="$HOME/Documents/Projects/MolSight"
PYTHON="/opt/homebrew/bin/python3.11"
CKPT_URL="https://huggingface.co/Robert-zwr/MolSight/resolve/main/pubchem_uspto_smiles_edges_30.pth?download=true"
CKPT_FILE="pubchem_uspto_smiles_edges_30.pth"

VENV_ONLY=false
if [[ "${1:-}" == "--venv" ]]; then
    VENV_ONLY=true
fi

# ── Check Python ──────────────────────────────────────────
if ! command -v "$PYTHON" &>/dev/null; then
    echo "ERROR: Python 3.11 not found at $PYTHON"
    echo "Install with: brew install python@3.11"
    exit 1
fi
echo "Using $($PYTHON --version)"

# ── Clone repo if needed ──────────────────────────────────
if [[ "$VENV_ONLY" == false ]]; then
    if [[ -d "$MOLSIGHT_DIR/.git" ]]; then
        echo "MolSight repo already exists at $MOLSIGHT_DIR"
        echo "Pulling latest..."
        cd "$MOLSIGHT_DIR" && git pull
    else
        echo "Cloning MolSight..."
        rm -rf "$MOLSIGHT_DIR"
        git clone https://github.com/hustvl/MolSight.git "$MOLSIGHT_DIR"
    fi
else
    if [[ ! -d "$MOLSIGHT_DIR" ]]; then
        echo "ERROR: MolSight directory not found at $MOLSIGHT_DIR"
        echo "Run without --venv for full setup."
        exit 1
    fi
fi

# ── Create venv ───────────────────────────────────────────
echo ""
echo "Creating venv..."
rm -rf "$MOLSIGHT_DIR/venv"
"$PYTHON" -m venv "$MOLSIGHT_DIR/venv"
"$MOLSIGHT_DIR/venv/bin/pip" install --upgrade pip -q

echo "Installing dependencies..."
"$MOLSIGHT_DIR/venv/bin/pip" install -q \
    "numpy<2" \
    torch \
    torchvision \
    timm \
    transformers \
    opencv-python-headless \
    Pillow \
    rdkit-pypi \
    "albumentations>=1.3.0,<2.0" \
    "epam.indigo" \
    SmilesPE \
    safetensors \
    scipy \
    pandas \
    openpyxl \
    datasets

echo "Dependencies installed."

# ── Download checkpoint ───────────────────────────────────
if [[ ! -f "$MOLSIGHT_DIR/$CKPT_FILE" ]]; then
    echo ""
    echo "Downloading pretrained checkpoint (~750 MB)..."
    "$MOLSIGHT_DIR/venv/bin/python" -c "
import urllib.request, os
url = '$CKPT_URL'
dst = '$MOLSIGHT_DIR/$CKPT_FILE'
urllib.request.urlretrieve(url, dst)
size = os.path.getsize(dst) / (1024*1024)
print(f'  Downloaded: {size:.0f} MB')
"
else
    echo ""
    echo "Checkpoint already exists, skipping download."
fi

# ── Verify ────────────────────────────────────────────────
echo ""
echo "Verifying installation..."
"$MOLSIGHT_DIR/venv/bin/python" -c "
import sys, os
os.chdir('$MOLSIGHT_DIR')
sys.path.insert(0, '$MOLSIGHT_DIR')
import torch
from molsight.model import MolsightModel
from molsight.tokenizer import CharTokenizer
print(f'  torch {torch.__version__}')
print(f'  Model imports OK')
print(f'  Checkpoint exists: {os.path.exists(\"$CKPT_FILE\")}')
"

echo ""
echo "MolSight setup complete at $MOLSIGHT_DIR"
