#!/usr/bin/env bash
# Launch pdf_to_smiles on macOS
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/venv/bin/activate"
python -m pdf_to_smiles "$@"
