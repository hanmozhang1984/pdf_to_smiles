@echo off
REM Build script for PDF to SMILES (Windows)
REM
REM Prerequisites:
REM   1. Python 3.9+ with pip
REM   2. Tesseract OCR installed (default: C:\Program Files\Tesseract-OCR)
REM   3. Virtual environment activated
REM
REM Usage:
REM   build.bat         - Build the application
REM   build.bat clean   - Clean build artifacts
REM   build.bat install - Install build dependencies

setlocal enabledelayedexpansion

set PROJECT_DIR=%~dp0
set DIST_DIR=%PROJECT_DIR%dist
set BUILD_DIR=%PROJECT_DIR%build

if "%1"=="clean" goto clean
if "%1"=="install" goto install
goto build

:install
echo Installing build dependencies...
pip install pyinstaller
pip install -r requirements.txt
goto end

:clean
echo Cleaning build artifacts...
if exist "%BUILD_DIR%" rmdir /s /q "%BUILD_DIR%"
if exist "%DIST_DIR%" rmdir /s /q "%DIST_DIR%"
if exist "%PROJECT_DIR%*.spec.bak" del /q "%PROJECT_DIR%*.spec.bak"
echo Clean complete.
goto end

:build
echo.
echo ============================================
echo  PDF to SMILES - Build Script (Windows)
echo ============================================
echo.

REM Check for Tesseract
if not exist "C:\Program Files\Tesseract-OCR\tesseract.exe" (
    echo WARNING: Tesseract not found at default location.
    echo Set TESSERACT_PATH environment variable if installed elsewhere.
    echo.
)

REM Check for virtual environment
if not defined VIRTUAL_ENV (
    echo WARNING: No virtual environment detected.
    echo It's recommended to build within the project's venv.
    echo.
)

echo Building application with PyInstaller...
echo.

pyinstaller --clean pdf_to_smiles.spec

if %ERRORLEVEL% neq 0 (
    echo.
    echo Build FAILED!
    exit /b 1
)

echo.
echo ============================================
echo  Build complete!
echo ============================================
echo.
echo Output directory: %DIST_DIR%\PDF-to-SMILES
echo.
echo To run the application:
echo   %DIST_DIR%\PDF-to-SMILES\PDF-to-SMILES.exe
echo.

:end
endlocal
