@echo off
title SIRIUS CMD-Gen Frontend
cd /d "%~dp0"
set "SIRIUS_PYTHON=C:\Users\hanut sai\miniconda3\envs\diffphar-fm\python.exe"
if not exist "%SIRIUS_PYTHON%" (
  echo Could not find the project Python environment:
  echo %SIRIUS_PYTHON%
  pause
  exit /b 1
)
echo Starting SIRIUS and the CMD-Gen API at http://127.0.0.1:8000
echo Keep this window open while using the frontend. Press Ctrl+C to stop.
start "" http://127.0.0.1:8000
"%SIRIUS_PYTHON%" sirius_server.py
pause
