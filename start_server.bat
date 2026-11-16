@echo off
title GANDIVA / CMD-GEN Selective Inhibitor Design Platform
echo ========================================================
echo   Starting GANDIVA / CMD-GEN Platform Server (Node.js)
echo   Target-Configurable Ensemble-Aware Pharmacophore & ML
echo ========================================================

start "" http://localhost:3000
agy-node server.js
pause
