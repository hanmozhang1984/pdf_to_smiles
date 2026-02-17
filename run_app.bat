@echo off
REM Launch PDF to SMILES on Windows
cd /d "%~dp0"
call venv\Scripts\activate.bat
python -m pdf_to_smiles
pause
