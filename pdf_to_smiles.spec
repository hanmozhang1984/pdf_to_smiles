# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for PDF to SMILES application.

This spec file bundles:
- All Python dependencies (TensorFlow, DECIMER, RDKit, etc.)
- Tesseract OCR binaries and language data
- Application resources

Build instructions:
    Windows: pyinstaller pdf_to_smiles.spec
    macOS:   pyinstaller pdf_to_smiles.spec
    Linux:   pyinstaller pdf_to_smiles.spec

Prerequisites:
    1. Install Tesseract OCR on your build machine
    2. Set TESSERACT_PATH environment variable (or use default paths)
"""

import os
import sys
from pathlib import Path

# Determine Tesseract paths based on platform
if sys.platform == 'win32':
    TESSERACT_DIR = os.environ.get('TESSERACT_PATH', r'C:\Program Files\Tesseract-OCR')
    TESSERACT_EXE = os.path.join(TESSERACT_DIR, 'tesseract.exe')
    TESSERACT_DLLS = [
        (os.path.join(TESSERACT_DIR, '*.dll'), 'tesseract'),
    ]
elif sys.platform == 'darwin':
    # macOS - Homebrew installation
    TESSERACT_DIR = os.environ.get('TESSERACT_PATH', '/opt/homebrew')
    if not os.path.exists(os.path.join(TESSERACT_DIR, 'bin', 'tesseract')):
        TESSERACT_DIR = '/usr/local'  # Intel Mac fallback
    TESSERACT_EXE = os.path.join(TESSERACT_DIR, 'bin', 'tesseract')
    TESSERACT_DLLS = []
else:
    # Linux
    TESSERACT_DIR = os.environ.get('TESSERACT_PATH', '/usr')
    TESSERACT_EXE = os.path.join(TESSERACT_DIR, 'bin', 'tesseract')
    TESSERACT_DLLS = []

# Tesseract data files (language models)
if sys.platform == 'win32':
    TESSDATA_DIR = os.path.join(TESSERACT_DIR, 'tessdata')
elif sys.platform == 'darwin':
    TESSDATA_DIR = os.path.join(TESSERACT_DIR, 'share', 'tessdata')
else:
    TESSDATA_DIR = '/usr/share/tesseract-ocr/4.00/tessdata'
    if not os.path.exists(TESSDATA_DIR):
        TESSDATA_DIR = '/usr/share/tessdata'

# Collect Tesseract binaries and data
tesseract_binaries = []
tesseract_datas = []

if os.path.exists(TESSERACT_EXE):
    tesseract_binaries.append((TESSERACT_EXE, 'tesseract'))
    print(f"Including Tesseract from: {TESSERACT_EXE}")
else:
    print(f"WARNING: Tesseract not found at {TESSERACT_EXE}")
    print("Set TESSERACT_PATH environment variable to your Tesseract installation")

# Add Windows DLLs
for pattern, dest in TESSERACT_DLLS if sys.platform == 'win32' else []:
    import glob
    for dll in glob.glob(pattern):
        tesseract_binaries.append((dll, dest))

# Add tessdata (language files) - only eng.traineddata for smaller bundle
if os.path.exists(TESSDATA_DIR):
    eng_data = os.path.join(TESSDATA_DIR, 'eng.traineddata')
    if os.path.exists(eng_data):
        tesseract_datas.append((eng_data, 'tesseract/tessdata'))
        print(f"Including tessdata from: {TESSDATA_DIR}")
    # Also include osd for orientation detection
    osd_data = os.path.join(TESSDATA_DIR, 'osd.traineddata')
    if os.path.exists(osd_data):
        tesseract_datas.append((osd_data, 'tesseract/tessdata'))
else:
    print(f"WARNING: tessdata not found at {TESSDATA_DIR}")

block_cipher = None

a = Analysis(
    ['src/pdf_to_smiles/main.py'],
    pathex=[],
    binaries=tesseract_binaries,
    datas=tesseract_datas,
    hiddenimports=[
        # TensorFlow related
        'tensorflow',
        'tensorflow.python',
        'tensorflow.python.eager',
        'keras',
        'keras.layers',
        'keras.models',
        'h5py',
        # DECIMER
        'decimer',
        'decimer.DECIMER',
        'decimer_segmentation',
        # RDKit
        'rdkit',
        'rdkit.Chem',
        'rdkit.Chem.Draw',
        'rdkit.Chem.AllChem',
        'rdkit.Chem.Descriptors',
        # Image processing
        'PIL',
        'PIL.Image',
        'cv2',
        'skimage',
        # PDF processing
        'pypdfium2',
        'pdfplumber',
        'pdfminer',
        'pdfminer.six',
        # GUI
        'PySide6',
        'PySide6.QtWidgets',
        'PySide6.QtCore',
        'PySide6.QtGui',
        # OCR
        'pytesseract',
        # Data
        'pandas',
        'numpy',
        # Other
        'efficientnet',
        'selfies',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary modules to reduce size
        'tkinter',
        'matplotlib.backends.backend_tkagg',
        'IPython',
        'jupyter',
        'notebook',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='PDF-to-SMILES',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # GUI application, no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add 'assets/icon.ico' if you have an icon
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='PDF-to-SMILES',
)

# macOS app bundle (only on macOS)
if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='PDF-to-SMILES.app',
        icon=None,  # Add 'assets/icon.icns' if you have an icon
        bundle_identifier='com.chemcipher.pdf-to-smiles',
        info_plist={
            'CFBundleName': 'PDF to SMILES',
            'CFBundleDisplayName': 'PDF to SMILES',
            'CFBundleVersion': '1.0.0',
            'CFBundleShortVersionString': '1.0.0',
            'NSHighResolutionCapable': True,
            'NSRequiresAquaSystemAppearance': False,  # Support dark mode
        },
    )
