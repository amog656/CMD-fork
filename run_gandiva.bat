@echo off
title GANDIVA - Target-Configurable Selective Inhibitor Design Platform
cd /d "%~dp0"
echo ====================================================================
echo   Starting GANDIVA (PharmFlow + GCPG + AutoDock Vina + WebGL UI)
echo ====================================================================

where python >nul 2>nul
if %ERRORLEVEL% equ 0 (
    set PYTHON_EXE=python
) else if exist "C:\Users\hanut sai\AppData\Local\Programs\Python\Python311\python.exe" (
    set PYTHON_EXE="C:\Users\hanut sai\AppData\Local\Programs\Python\Python311\python.exe"
) else (
    set PYTHON_EXE=py
)

start "" http://localhost:3000
%PYTHON_EXE% -m pharmpipe.web_bridge 3000
pause
