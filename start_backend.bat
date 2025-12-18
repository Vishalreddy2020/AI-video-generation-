@echo off
echo Starting AI Video Generator Backend...
cd backend
if not exist venv (
    echo Virtual environment not found. Please run setup.bat first!
    pause
    exit /b 1
)
call venv\Scripts\activate
echo Starting server...
python main.py
pause

