@echo off
echo ========================================
echo AI Video Generator - Setup Checker
echo ========================================
echo.

echo [1/5] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo [X] Python is not installed or not in PATH
    echo     Please install Python 3.8 or higher
) else (
    python --version
    echo [OK] Python is installed
)
echo.

echo [2/5] Checking Node.js installation...
node --version >nul 2>&1
if errorlevel 1 (
    echo [X] Node.js is not installed or not in PATH
    echo     Please install Node.js 16 or higher
) else (
    node --version
    echo [OK] Node.js is installed
)
echo.

echo [3/5] Checking backend virtual environment...
if exist backend\venv (
    echo [OK] Backend virtual environment exists
) else (
    echo [X] Backend virtual environment not found
    echo     Run: cd backend ^&^& setup.bat
)
echo.

echo [4/5] Checking backend dependencies...
if exist backend\venv (
    call backend\venv\Scripts\activate
    python -c "import fastapi" >nul 2>&1
    if errorlevel 1 (
        echo [X] Backend dependencies not installed
        echo     Run: cd backend ^&^& setup.bat
    ) else (
        echo [OK] Backend dependencies installed
    )
    call backend\venv\Scripts\deactivate
) else (
    echo [SKIP] Cannot check - venv not found
)
echo.

echo [5/5] Checking frontend dependencies...
if exist frontend\node_modules (
    echo [OK] Frontend dependencies installed
) else (
    echo [X] Frontend dependencies not installed
    echo     Run: cd frontend ^&^& npm install
)
echo.

echo ========================================
echo Setup Check Complete
echo ========================================
echo.
echo To start the application:
echo   1. Terminal 1: start_backend.bat
echo   2. Terminal 2: start_frontend.bat
echo.
pause

