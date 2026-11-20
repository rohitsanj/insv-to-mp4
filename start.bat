@echo off
rem Launch 360 Views on Windows. First run sets up a private Python environment in .venv.
cd /d "%~dp0"
where py >nul 2>nul && (set PY=py -3) || (set PY=python)
if not exist .venv\Scripts\python.exe (
  echo First run: setting up ^(one time^)...
  %PY% -m venv .venv || (echo 360 Views needs Python 3.9+ from https://www.python.org/downloads/ & pause & exit /b 1)
)
fc /b requirements.txt .venv\requirements.stamp >nul 2>nul || (
  .venv\Scripts\python -m pip install --disable-pip-version-check -q -r requirements.txt && copy /y requirements.txt .venv\requirements.stamp >nul
  .venv\Scripts\python -m views360 doctor
)
where ffmpeg >nul 2>nul || (echo 360 Views needs ffmpeg:  winget install Gyan.FFmpeg   then open a new window. & pause & exit /b 1)
.venv\Scripts\python -m views360 %*
pause
