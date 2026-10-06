@echo off
REM Sets PY to the Python the numbered .bat files run with. Called by them, not run alone.
REM
REM The project's own .venv first. The shared one in D:\pill-counter-lite is where they all
REM used to point, and is only tried if .venv is missing -- a machine without that folder
REM got nothing but "The system cannot find the path specified." and no hint of what was
REM missing.
set "PY=%~dp0.venv\Scripts\python.exe"
if exist "%PY%" exit /b 0
set "PY=D:\pill-counter-lite\pillcount-v12\.venv-train\Scripts\python.exe"
if exist "%PY%" exit /b 0

echo.
echo  No Python environment found. Expected: %~dp0.venv
echo.
echo  Create it once, from this folder (Python 3.10):
echo    python -m venv .venv
echo    .venv\Scripts\python.exe -m pip install PySide6 opencv-python "numpy<2" onnxruntime
echo    .venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
echo    .venv\Scripts\python.exe -m pip install ultralytics
echo.
pause
exit /b 1
