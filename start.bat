@echo off
title SIRIUS Local Server
echo ==========================================
echo Starting SIRIUS Local Server on http://localhost:8000 ...
echo Press Ctrl+C to stop.
echo ==========================================
python -m http.server 8000
pause
