#!/usr/bin/env bash
# Launch PDF to SMILES on macOS/Linux
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/venv/bin/activate"
python -m pdf_to_smiles "$@"
