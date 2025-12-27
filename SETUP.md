# Setup Guide

Complete setup instructions for the AI Video Generator.

## Initial Setup

### Step 1: Backend Setup

**Windows (Recommended):**
```bash
cd backend
setup.bat
```

**Linux/Mac:**
```bash
cd backend
bash setup.sh
```

**Manual Setup:**
```bash
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
# or: source venv/bin/activate  # Linux/Mac
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Step 2: Frontend Setup

```bash
cd frontend
npm install
```

## Running the Application

### Start Backend

```bash
cd backend
venv\Scripts\activate  # Windows
# or: source venv/bin/activate  # Linux/Mac
python main.py
```

Backend will be available at: http://localhost:8000

### Start Frontend

Open a new terminal:
```bash
cd frontend
npm start
```

Frontend will be available at: http://localhost:3000

## Optional: Face Recognition

For enhanced face replacement accuracy:

**Windows:**
```bash
cd backend
install_face_recognition.bat
```

**Linux/Mac:**
```bash
cd backend
bash install_face_recognition.sh
```

**Manual Installation:**
```bash
# Windows - Pre-built wheel (easiest)
pip install dlib-binary
pip install face-recognition

# Or using conda
conda install -c conda-forge dlib
pip install face-recognition

# Linux/Mac
pip install cmake
pip install dlib
pip install face-recognition
```

Note: Face replacement works without this using OpenCV's built-in detection, but with lower accuracy.

## Optional: AI Model Setup

For advanced AI video generation features:

**Windows:**
```bash
cd backend
install_universal_ai.bat
```

**Manual Installation:**
```bash
cd backend
venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install diffusers transformers accelerate safetensors
```

This enables:
- Image generation from text prompts
- Image editing with AI
- Advanced video generation

Models will download automatically on first use (~4-8GB).

## Optional: Intel Arc GPU Setup (OpenVINO)

For Intel Arc GPU optimization (fastest path):

**Install OpenVINO (Recommended):**
```bash
cd backend
venv\Scripts\activate
pip install optimum[openvino]
```

**Alternative (if OpenVINO not available):**
```bash
pip install diffusers transformers accelerate
pip install intel-extension-for-pytorch
```

**Test Image Generation:**
```bash
# Start backend
python main.py

# Test endpoint (in another terminal)
curl -X POST "http://localhost:8000/image/generate" \
  -F "prompt=a cat wearing sunglasses" \
  -F "width=512" \
  -F "height=512" \
  -o test_image.png
```

**Performance with Intel Arc + OpenVINO:**
- First generation: 30-60 seconds (model loading)
- Subsequent generations: 5-15 seconds per image

**Performance with Intel Arc + Standard Diffusers:**
- First generation: 1-2 minutes (model loading)
- Subsequent generations: 10-30 seconds per image

## System Requirements

**Minimum:**
- Python 3.8+
- Node.js 16+
- 8GB RAM
- 10GB free disk space

**Recommended:**
- Python 3.10+
- Node.js 18+
- 16GB+ RAM
- 20GB+ free disk space
- GPU (NVIDIA, Intel Arc, or Apple Silicon) for faster processing

## Verification

After setup, verify everything works:

1. Test backend: http://localhost:8000/api/health
   - Should return: `{"status": "healthy"}`

2. Test frontend: http://localhost:3000
   - Should show the application interface

3. Try generating a test video:
   - Upload a photo or enter a text prompt
   - Click "Generate Video"
   - Wait for processing to complete

## Troubleshooting

See README.md for detailed troubleshooting steps.

Common issues:
- Setuptools errors: Run `python -m pip install --upgrade pip setuptools wheel`
- Port conflicts: Change ports in configuration files
- Module not found: Re-run setup scripts
- Connection issues: Verify both servers are running

