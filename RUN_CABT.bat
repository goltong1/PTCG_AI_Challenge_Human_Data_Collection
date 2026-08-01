@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "CABT_PYTHON=%~dp0runtime\python\python.exe"
set "CABT_READY=%~dp0runtime\.cabt-ready-v2"

if not exist "%CABT_PYTHON%" goto setup_runtime
if not exist "%CABT_READY%" goto setup_runtime
goto run_server

:setup_runtime
echo.
echo [CABT] Preparing the portable runtime. This is required only once.
echo [CABT] No system-wide Python installation will be created.
echo.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\bootstrap.ps1"
if errorlevel 1 goto setup_failed

:run_server
echo.
echo [CABT] Starting Human vs AI Arena and Replay Viewer...
echo [CABT] Open http://127.0.0.1:8765 if the browser does not open.
echo.
"%CABT_PYTHON%" "%~dp0app\server.py" --host 127.0.0.1 --port 8765
exit /b %errorlevel%

:setup_failed
echo.
echo [CABT] Runtime setup failed.
echo [CABT] Check your internet connection and run this file again.
echo.
pause
exit /b 1
