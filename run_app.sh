#!/bin/bash
# Launch script for Mac/Linux

cd "$(dirname "$0")/src"
source ../venv/bin/activate
python -m pdf_to_smiles.main
