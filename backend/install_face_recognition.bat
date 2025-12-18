@echo off
echo Installing Face Recognition Dependencies...
echo.
echo This script will install face_recognition and dlib for Windows.
echo.
echo Method 1: Using pre-built wheels (Recommended for Windows)
echo ============================================================

cd backend
if not exist venv (
    echo ERROR: Virtual environment not found. Please run setup.bat first!
    pause
    exit /b 1
)

call venv\Scripts\activate

echo.
echo Installing dlib-binary (pre-built wheel, easier on Windows)...
pip install dlib-binary
if errorlevel 1 (
    echo.
    echo dlib-binary installation failed. Trying alternative method...
    echo.
    echo Method 2: Installing from source (may take longer)
    echo ===================================================
    echo Installing CMake first...
    pip install cmake
    echo Installing dlib...
    pip install dlib
)

echo.
echo Installing face-recognition...
pip install face-recognition

echo.
echo ========================================
echo Installation completed!
echo ========================================
echo.
echo Face replacement feature should now work.
echo Restart the backend server to use it.
echo.
pause

