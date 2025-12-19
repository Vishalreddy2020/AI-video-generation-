# AI Video Generator

An AI-powered application that generates videos from photos, text prompts, or existing videos. Features face replacement capabilities and customizable video duration (5 or 10 seconds) with 720p resolution.

## Features

- Photo to Video: Transform static images into animated videos
- Text to Video: Generate videos from text descriptions
- Video Processing: Enhance and process existing videos
- Face Replacement: Replace faces in generated videos with your own
- Duration Options: Choose between 5 or 10 second videos
- 720p Resolution: High-quality video output

## Project Structure

```
AI-video-generation-/
├── backend/              # FastAPI backend
│   ├── main.py          # API endpoints
│   ├── services/        # Video generation and face swapping services
│   └── requirements.txt # Python dependencies
├── frontend/            # React frontend
│   ├── src/
│   │   ├── components/  # React components
│   │   └── App.js       # Main app component
│   └── package.json     # Node dependencies
└── README.md
```

## Prerequisites

- Python 3.8 or higher
- Node.js 16 or higher
- FFmpeg (for video processing, optional)
- CUDA-capable GPU (optional, for faster processing)

## Quick Start

### 1. Backend Setup

**Windows:**
```bash
cd backend
setup.bat
```

**Linux/Mac:**
```bash
cd backend
bash setup.sh
```

The setup script will:
- Create a virtual environment
- Upgrade pip, setuptools, and wheel
- Install all core dependencies

### 2. Frontend Setup

```bash
cd frontend
npm install
```

### 3. Run the Application

**Start Backend (Terminal 1):**
```bash
cd backend
venv\Scripts\activate  # Windows
# or: source venv/bin/activate  # Linux/Mac
python main.py
```

**Start Frontend (Terminal 2):**
```bash
cd frontend
npm start
```

The application will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000

## Installation Details

### Backend Manual Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
- Windows: `venv\Scripts\activate`
- Linux/Mac: `source venv/bin/activate`

4. Upgrade pip, setuptools, and wheel first:
```bash
python -m pip install --upgrade pip setuptools wheel
```

5. Install core dependencies:
```bash
pip install -r requirements.txt
```

### Optional: Face Recognition Setup

Face replacement works with OpenCV's built-in face detection by default (no additional installation needed). For better accuracy, you can install face_recognition:

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

Note: Face replacement will work without face_recognition using OpenCV's Haar Cascade detector, but face_recognition provides better accuracy.

## Usage

1. Upload Input: Choose to upload a photo/video OR enter a text prompt
2. Set Duration: Select 5 or 10 seconds for your video
3. Face Replacement (Optional): 
   - Enable "Replace face in video"
   - Upload your face image
4. Generate: Click "Generate Video" and wait for processing
5. Download: Preview and download your generated video

## API Endpoints

- `POST /api/generate-video`: Generate video from input
  - Parameters:
    - `file`: Photo or video file (optional)
    - `text_prompt`: Text description (optional)
    - `duration`: 5 or 10 (seconds)
    - `replace_face`: Boolean
    - `face_image`: Face image file (if replace_face is true)

- `GET /api/health`: Health check endpoint

## Technical Details

- Video Resolution: 1280x720 (720p)
- Frame Rate: 24 fps
- Video Format: MP4 (H.264)
- Face Detection: Uses face_recognition library (optional, OpenCV fallback available)
- Video Processing: OpenCV and FFmpeg

## Troubleshooting

### Setuptools/Build Errors

If you see "Cannot import 'setuptools.build_meta'" or similar errors:

1. Make sure you've upgraded pip, setuptools, and wheel:
   ```bash
   python -m pip install --upgrade pip setuptools wheel
   ```

2. Then try installing requirements again:
   ```bash
   pip install -r requirements.txt
   ```

3. If issues persist, try recreating the virtual environment

### dlib Installation Issues

If you encounter issues installing dlib:
- Windows: Use conda: `conda install -c conda-forge dlib`
- Windows Alternative: Use pre-built wheel: `pip install dlib-binary`
- Linux/Mac: Install CMake first: `pip install cmake` or `sudo apt-get install cmake`
- Face recognition will work without dlib, but with limited functionality

### FFmpeg Not Found

Install FFmpeg:
- Windows: Download from https://ffmpeg.org/download.html and add to PATH
- Linux: `sudo apt-get install ffmpeg`
- Mac: `brew install ffmpeg`

Note: FFmpeg is optional but recommended for better video encoding. The app will work without it but may produce larger files.

### Connection Issues

If the frontend cannot connect to the backend:

1. Verify backend is running: Check http://localhost:8000/api/health
2. Check CORS settings in `backend/main.py`
3. Ensure both servers are running on correct ports
4. Check browser console (F12) for error messages

### Port Already in Use

- Backend: Change port in `backend/main.py` (line 121)
- Frontend: React will automatically use the next available port

## License

This project is open source and available for modification and distribution.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
