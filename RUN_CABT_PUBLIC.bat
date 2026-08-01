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
set "CABT_PUBLIC_MODE=1"
set "CABT_ENABLE_RESULT_SUBMISSION=1"
set "CABT_ENABLE_ONLINE_MATCHING=1"
set "CABT_ADMIN_TOKEN=change-this-admin-token"
set "CABT_MAX_SESSIONS=4"
set "CABT_MAX_PVP_MATCHES=4"
set "CABT_MAX_PVP_WAITERS=32"
set "CABT_SESSION_IDLE_SECONDS=3600"
set "CABT_PVP_QUEUE_TIMEOUT=600"
set "CABT_PVP_MATCH_IDLE_SECONDS=7200"
set "CABT_DATA_DIR=%~dp0server_data"

echo.
echo [CABT] Starting public Online Battle Arena on port 8765...
echo [CABT] Arena: http://127.0.0.1:8765
echo [CABT] Admin: http://127.0.0.1:8765/admin
echo [CABT] Change CABT_ADMIN_TOKEN before exposing this server to the internet.
echo.
"%CABT_PYTHON%" "%~dp0app\server.py" --host 0.0.0.0 --port 8765
exit /b %errorlevel%

:setup_failed
echo.
echo [CABT] Runtime setup failed.
echo [CABT] Check your internet connection and run this file again.
echo.
pause
exit /b 1
