@echo off
title DerivX 4D - Universal Symbolic Derivative Engine
echo ============================================================
echo   Starting DerivX 4D (Symbolic Engine & Visualizer)
echo ============================================================
echo.
call .\venv\Scripts\activate.bat
streamlit run app.py
pause
