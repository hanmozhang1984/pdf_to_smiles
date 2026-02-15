"""Path utilities for finding bundled resources in frozen applications."""

from __future__ import annotations

import os
import sys
import shutil


def get_app_dir() -> str:
    """Get the application directory, handling both frozen and development modes."""
    if getattr(sys, 'frozen', False):
        # Running as compiled executable (PyInstaller)
        return sys._MEIPASS
    else:
        # Running in development
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_tesseract_cmd() -> str | None:
    """
    Find the Tesseract executable, checking bundled location first.

    Returns:
        Path to tesseract executable, or None if not found.
    """
    # Check for bundled Tesseract first (frozen app)
    if getattr(sys, 'frozen', False):
        app_dir = sys._MEIPASS
        if sys.platform == 'win32':
            bundled_path = os.path.join(app_dir, 'tesseract', 'tesseract.exe')
        else:
            bundled_path = os.path.join(app_dir, 'tesseract', 'tesseract')

        if os.path.exists(bundled_path):
            return bundled_path

    # Check standard installation paths
    if sys.platform == 'win32':
        standard_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        ]
    elif sys.platform == 'darwin':
        standard_paths = [
            '/usr/local/bin/tesseract',      # Intel Mac (Homebrew)
            '/opt/homebrew/bin/tesseract',   # Apple Silicon Mac (Homebrew)
        ]
    else:
        standard_paths = [
            '/usr/bin/tesseract',
            '/usr/local/bin/tesseract',
        ]

    for path in standard_paths:
        if os.path.exists(path):
            return path

    # Fall back to PATH lookup
    found = shutil.which('tesseract')
    if found:
        return found

    return None


def get_tessdata_dir() -> str | None:
    """
    Find the Tesseract data directory.

    Returns:
        Path to tessdata directory, or None if not found.
    """
    # Check for bundled tessdata first (frozen app)
    if getattr(sys, 'frozen', False):
        app_dir = sys._MEIPASS
        bundled_path = os.path.join(app_dir, 'tesseract', 'tessdata')
        if os.path.exists(bundled_path):
            return bundled_path

    # Check environment variable
    tessdata_prefix = os.environ.get('TESSDATA_PREFIX')
    if tessdata_prefix and os.path.exists(tessdata_prefix):
        return tessdata_prefix

    # Check standard locations
    if sys.platform == 'win32':
        standard_paths = [
            r'C:\Program Files\Tesseract-OCR\tessdata',
            r'C:\Program Files (x86)\Tesseract-OCR\tessdata',
        ]
    elif sys.platform == 'darwin':
        standard_paths = [
            '/usr/local/share/tessdata',
            '/opt/homebrew/share/tessdata',
        ]
    else:
        standard_paths = [
            '/usr/share/tesseract-ocr/4.00/tessdata',
            '/usr/share/tesseract-ocr/5/tessdata',
            '/usr/share/tessdata',
        ]

    for path in standard_paths:
        if os.path.exists(path):
            return path

    return None


def configure_tesseract():
    """
    Configure pytesseract to use the correct Tesseract executable.

    Call this function early in your application startup.

    Returns:
        True if Tesseract was found and configured, False otherwise.
    """
    try:
        import pytesseract
    except ImportError:
        return False

    tesseract_cmd = get_tesseract_cmd()
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        # Also set TESSDATA_PREFIX if we found it
        tessdata_dir = get_tessdata_dir()
        if tessdata_dir:
            os.environ['TESSDATA_PREFIX'] = tessdata_dir

        return True

    return False
