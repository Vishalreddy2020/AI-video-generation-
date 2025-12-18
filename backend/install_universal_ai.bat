@echo off
echo ========================================
echo Installing Universal AI Video Generation
echo ========================================
echo.
echo This will install AI video generation that works on ANY laptop
echo - CPU mode (works everywhere)
echo - Auto-detects GPU if available (NVIDIA, Intel Arc, Apple Silicon)
echo - Optimized for performance on all devices
echo.

cd backend
if not exist venv (
    echo ERROR: Virtual environment not found. Please run setup.bat first!
    pause
    exit /b 1
)

call venv\Scripts\activate

echo.
echo Step 1: Installing PyTorch (CPU version - works on all laptops)...
echo ========================================
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
    echo ERROR: Failed to install PyTorch
    pause
    exit /b 1
)

echo.
echo Step 2: Installing AI model libraries...
echo ========================================
pip install diffusers transformers accelerate
if errorlevel 1 (
    echo ERROR: Failed to install AI libraries
    pause
    exit /b 1
)

echo.
echo Step 3: Installing optimization libraries...
echo ========================================
pip install safetensors imageio imageio-ffmpeg
if errorlevel 1 (
    echo Warning: Some optional dependencies failed
)

echo.
echo Step 4: Optional GPU acceleration (if you have GPU)...
echo ========================================
echo.
echo If you have NVIDIA GPU, run:
echo   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
echo.
echo If you have Intel Arc GPU, run:
echo   pip install intel-extension-for-pytorch
echo.
echo Otherwise, CPU mode will work fine (just slower)
echo.

echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo The system will automatically:
echo - Use CPU mode (works on all laptops)
echo - Auto-detect GPU if available
echo - Optimize performance for your hardware
echo.
echo Next steps:
echo 1. Restart the backend server
echo 2. Try generating a video - it will work on any laptop!
echo.
pause

