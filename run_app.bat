@echo off
cd /d C:\Users\HanmoZhang\Documents\Projects\todo\pdf_to_smiles\src
call ..\venv\Scripts\activate.bat
python -m pdf_to_smiles.main
pause
