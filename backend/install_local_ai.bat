@echo off
echo ========================================
echo Installing Local AI Video Generation
echo ========================================
echo.
echo This will install all dependencies needed to run
echo AI video generation models locally (no API keys needed!)
echo.
echo Note: This requires significant disk space (~10-15GB for models)
echo       and a GPU is highly recommended for reasonable speed.
echo.

cd backend
if not exist venv (
    echo ERROR: Virtual environment not found. Please run setup.bat first!
    pause
    exit /b 1
)

call venv\Scripts\activate

echo.
echo Step 1: Installing core ML libraries...
echo ========================================
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
if errorlevel 1 (
    echo Warning: CUDA version failed, trying CPU version...
    pip install torch torchvision torchaudio
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
echo Step 3: Installing additional dependencies...
echo ========================================
pip install safetensors imageio imageio-ffmpeg
if errorlevel 1 (
    echo Warning: Some optional dependencies failed, but core should work
)

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Restart the backend server
echo 2. On first video generation, the model will download automatically (~5GB)
echo 3. This may take 10-20 minutes on first run
echo.
echo GPU Requirements:
echo - NVIDIA GPU with 8GB+ VRAM recommended
echo - Will work on CPU but will be very slow (10-30 minutes per video)
echo.
pause

