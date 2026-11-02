@echo off
title Student Academic Performance Analytics - Presentation Launcher
echo ================================================================
echo   Starting Student Academic Performance Analytics Dashboard
echo ================================================================
echo.
echo 1. Starting Streamlit server...
start /b "" "C:\Users\keerthi\AppData\Local\Python\pythoncore-3.11-64\Scripts\streamlit.exe" run app.py --server.port 8501 --server.headless false
timeout /t 3 >nul
echo 2. Opening local browser: http://localhost:8501
start http://localhost:8501
echo.
echo Dashboard is running! Keep this window open during your presentation.
pause
