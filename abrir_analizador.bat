@echo off
cd /d "C:\Users\ANDRES.TORRES\Downloads\pliegos_ai_web\pliegos_ai_web\backend"
start http://127.0.0.1:8000
call python -m uvicorn main:app --reload
pause
