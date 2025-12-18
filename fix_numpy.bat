@echo off
echo Fixing NumPy compatibility issue...
echo.
echo OpenCV requires NumPy 1.x, but NumPy 2.x is installed.
echo Downgrading NumPy to compatible version...
echo.

cd backend
if not exist venv (
    echo ERROR: Virtual environment not found. Please run setup.bat first!
    pause
    exit /b 1
)

call venv\Scripts\activate

echo Uninstalling incompatible packages...
pip uninstall numpy opencv-python -y

echo Installing compatible versions...
echo Upgrading OpenCV to version compatible with NumPy 2.x...
pip install opencv-python-headless
pip install numpy

echo.
echo Verifying installation...
python -c "import numpy; print('NumPy version:', numpy.__version__)"
python -c "import cv2; print('OpenCV version:', cv2.__version__)"

echo.
echo ========================================
echo NumPy fix completed!
echo ========================================
echo.
echo You can now run: python main.py
echo.
pause

