@echo off
title GANDIVA - Target-Configurable Selective Inhibitor Design Platform
cd /d "%~dp0\..\.."
echo ====================================================================
echo   Starting GANDIVA (PharmFlow + GCPG + AutoDock Vina + WebGL UI)
echo ====================================================================

start "" http://localhost:3000
"C:\Users\hanut sai\AppData\Local\Programs\Python\Python311\python.exe" -m pharmpipe.web_bridge 3000
pause
