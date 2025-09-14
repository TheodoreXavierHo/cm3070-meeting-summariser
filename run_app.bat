@echo off
setlocal EnableExtensions EnableDelayedExpansion
title CM3070 Summariser - launcher

REM ---- paths ----
set "VENV_DIR=.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "REQ_FILE=requirements.txt"
set "REQ_HASH_FILE=%VENV_DIR%\.req_hash"

echo [run_app] Working dir: %CD%

REM ---- 0) create venv if missing ----
if not exist "%VENV_PY%" (
  echo [run_app] Creating virtual environment...
  py -3 -m venv "%VENV_DIR%" 2>nul || (
    echo [run_app] 'py' not found; trying 'python'...
    python -m venv "%VENV_DIR%" || (
      echo [run_app][ERROR] Could not create virtualenv.
      exit /b 1
    )
  )
)

REM ---- 1) upgrade pip ----
echo [run_app] Upgrading pip...
"%VENV_PY%" -m pip install --upgrade pip wheel setuptools
if errorlevel 1 (
  echo [run_app][ERROR] pip upgrade failed.
  exit /b 1
)

REM ---- 2) install/refresh requirements on change ----
if not exist "%REQ_FILE%" (
  echo [run_app][ERROR] %REQ_FILE% not found.
  exit /b 1
)

REM compute hash of requirements.txt using certutil (available on Win10+)
for /f "tokens=1" %%H in ('certutil -hashfile "%REQ_FILE%" SHA256 ^| find /i /v "hash" ^| find /i /v "certutil"') do set "REQ_HASH=%%H"

set "NEED_REINSTALL=1"
if exist "%REQ_HASH_FILE%" (
  set /p OLD_HASH=<"%REQ_HASH_FILE%"
  if /i "!OLD_HASH!"=="!REQ_HASH!" set "NEED_REINSTALL=0"
)

if "!NEED_REINSTALL!"=="1" (
  echo [run_app] Installing Python deps from %REQ_FILE% ...
  "%VENV_PY%" -m pip install -r "%REQ_FILE%"
  if errorlevel 1 (
    echo [run_app][ERROR] pip install failed.
    exit /b 1
  )
  >"%REQ_HASH_FILE%" echo !REQ_HASH!
) else (
  echo [run_app] Requirements unchanged; skipping reinstall.
)

REM ---- 3) install Torch (CUDA if NVIDIA GPU present, else CPU) ----
REM     This script auto-detects GPU and selects proper wheels.
REM     Make sure the filename is tools\install_torch.py (NOT "install_touch.py").
if exist "tools\install_torch.py" (
  echo [run_app] Ensuring PyTorch is installed...
  "%VENV_PY%" "tools\install_torch.py"
  if errorlevel 1 (
    echo [run_app][WARN] Torch install reported an issue; continuing anyway.
  )
) else (
  echo [run_app][WARN] tools\install_torch.py not found; skipping Torch step.
)

REM ---- 4) launch Streamlit ----
if not exist "app\app.py" (
  echo [run_app][ERROR] app\app.py not found.
  exit /b 1
)

echo [run_app] Starting Streamlit...
"%VENV_PY%" -m streamlit run "app\app.py"
set "EC=%ERRORLEVEL%"
echo [run_app] Streamlit exited with code %EC%

endlocal & exit /b %EC%
