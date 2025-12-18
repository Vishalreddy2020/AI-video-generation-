@echo off
echo ========================================
echo Installing Intel Arc GPU Support
echo ========================================
echo.
echo This will install PyTorch with Intel Arc GPU support
echo and Intel Extension for PyTorch to use your Intel Arc GPU and NPU.
echo.

cd backend
if not exist venv (
    echo ERROR: Virtual environment not found. Please run setup.bat first!
    pause
    exit /b 1
)

call venv\Scripts\activate

echo.
echo Step 1: Installing PyTorch (CPU version - Intel Arc compatible)...
echo ========================================
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
    echo ERROR: Failed to install PyTorch
    pause
    exit /b 1
)

echo.
echo Step 2: Installing Intel Extension for PyTorch...
echo ========================================
echo This enables Intel Arc GPU and NPU acceleration
pip install intel-extension-for-pytorch
if errorlevel 1 (
    echo Warning: Intel Extension installation failed
    echo You can still use CPU mode, but GPU acceleration won't be available
)

echo.
echo Step 3: Installing AI model libraries...
echo ========================================
pip install diffusers transformers accelerate
if errorlevel 1 (
    echo ERROR: Failed to install AI libraries
    pause
    exit /b 1
)

echo.
echo Step 4: Installing additional dependencies...
echo ========================================
pip install safetensors imageio imageio-ffmpeg
if errorlevel 1 (
    echo Warning: Some optional dependencies failed
)

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Your Intel Arc GPU and NPU will be automatically detected
echo and used for video generation acceleration.
echo.
echo Next steps:
echo 1. Restart the backend server
echo 2. The system will automatically use Intel Arc GPU if available
echo.
pause

