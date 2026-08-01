@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo.
echo CABT GitHub Pages Publisher
echo Git for Windows must be installed and GitHub login must be configured.
echo.
set /p "REPO_URL=GitHub repository URL (example: https://github.com/user/cabt-local-arena.git): "
if "%REPO_URL%"=="" goto cancelled

where git >nul 2>nul
if errorlevel 1 (
  echo.
  echo Git was not found. Install Git for Windows or use GitHub Desktop.
  pause
  exit /b 1
)

if not exist ".git" git init

git add .
git commit -m "Deploy CABT static replay viewer"
git branch -M main

git remote get-url origin >nul 2>nul
if errorlevel 1 (
  git remote add origin "%REPO_URL%"
) else (
  git remote set-url origin "%REPO_URL%"
)

git push -u origin main
if errorlevel 1 (
  echo.
  echo Push failed. Check GitHub authentication and repository URL.
  pause
  exit /b 1
)

echo.
echo Upload complete.
echo GitHub: Settings - Pages - Deploy from a branch - main - root
pause
exit /b 0

:cancelled
echo Cancelled.
pause
