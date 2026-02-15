$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
Set-Location 'C:\Users\HanmoZhang\Documents\Projects\todo\pdf_to_smiles'
python -m modal deploy src/pdf_to_smiles/cloud/modal_app.py 2>&1 | Out-File -FilePath deploy_output.txt -Encoding utf8
