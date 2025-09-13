@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ---- paths ----
set VENV_DIR=.venv
set REQ_HASH_FILE=%VENV_DIR%\.req_hash
set REQ_HASH_CUR=%VENV_DIR%\.req_hash.cur

REM ---- 0) create venv once ----
if not exist "%VENV_DIR%" (
  echo [run_app] Creating virtual environment...
  python -m venv "%VENV_DIR%"
  if errorlevel 1 (
    echo [run_app] Failed to create venv. Is Python on PATH?
    exit /b 1
  )
)

REM ---- 1) activate venv ----
call "%VENV_DIR%\Scripts\activate"
if errorlevel 1 (
  echo [run_app] Failed to activate venv.
  exit /b 1
)

REM ---- 2) install/upgrade pip just once (optional) ----
if not exist "%VENV_DIR%\.pip_upgraded" (
  echo [run_app] Upgrading pip (one-time)...
  python -m pip install --upgrade pip
  if errorlevel 1 goto :pipfail
  echo done> "%VENV_DIR%\.pip_upgraded"
)

REM ---- 3) if NVIDIA GPU present and torch CUDA not active, fix torch FIRST ----
where nvidia-smi >nul 2>&1
if not errorlevel 1 (
  for /f %%I in ('python -c "import sys; \
try:\n import torch;print('ok' if getattr(torch,'cuda',None) and torch.cuda.is_available() else 'bad') \
\nexcept Exception:\n print('missing')"') do set TORCH_OK=%%I
  if not "%TORCH_OK%"=="ok" (
    echo [run_app] NVIDIA GPU detected, ensuring CUDA torch wheels...
    python tools\install_torch.py
    if errorlevel 1 (
      echo [run_app] Torch install/upgrade failed. Close Streamlit/terminals using the venv and run again.
      exit /b 1
    )
  ) else (
    echo [run_app] CUDA torch already active; skipping torch install.
  )
) else (
  echo [run_app] No NVIDIA GPU detected; skipping CUDA torch step.
)

REM ---- 4) install project requirements ONLY if requirements.txt changed ----
certutil -hashfile requirements.txt SHA256 > "%REQ_HASH_CUR%" 2>nul
if not exist "%REQ_HASH_FILE%" (
  set NEED_REQ=1
) else (
  fc /b "%REQ_HASH_FILE%" "%REQ_HASH_CUR%" >nul 2>&1
  if errorlevel 1 ( set NEED_REQ=1 ) else ( set NEED_REQ=0 )
)

if "%NEED_REQ%"=="1" (
  echo [run_app] Installing/updating Python deps from requirements.txt...
  python -m pip install -r requirements.txt
  if errorlevel 1 goto :pipfail
  move /Y "%REQ_HASH_CUR%" "%REQ_HASH_FILE%" >nul
) else (
  del "%REQ_HASH_CUR%" >nul 2>&1
  echo [run_app] Requirements unchanged; skipping pip install.
)

REM ---- 5) launch app ----
echo [run_app] Starting Streamlit...
python -m streamlit run app.py
goto :end

:pipfail
echo [run_app] pip failed (network or permissions). Fix and re-run.
exit /b 1

:end
endlocal
