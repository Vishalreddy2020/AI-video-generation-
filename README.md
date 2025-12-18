# AI Video Generator

An AI-powered application that generates videos from photos, text prompts, or existing videos. Features face replacement capabilities and customizable video duration (5 or 10 seconds) with 720p resolution.

## Features

- 📸 **Photo to Video**: Transform static images into animated videos
- 📝 **Text to Video**: Generate videos from text descriptions
- 🎬 **Video Processing**: Enhance and process existing videos
- 👤 **Face Replacement**: Replace faces in generated videos with your own
- ⏱️ **Duration Options**: Choose between 5 or 10 second videos
- 🎥 **720p Resolution**: High-quality video output

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
- FFmpeg (for video processing)
- CUDA-capable GPU (optional, for faster processing)

## Installation

### Backend Setup

**Quick Setup (Recommended):**

1. Navigate to the backend directory:
```bash
cd backend
```

2. Run the setup script:
- **Windows**: Double-click `setup.bat` or run it from command prompt
- **Linux/Mac**: Run `bash setup.sh` or `chmod +x setup.sh && ./setup.sh`

The setup script will:
- Create a virtual environment
- Upgrade pip, setuptools, and wheel
- Install all core dependencies

**Manual Setup:**

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

4. **Important**: Upgrade pip, setuptools, and wheel first:
```bash
python -m pip install --upgrade pip setuptools wheel
```

5. Install core dependencies:
```bash
pip install -r requirements.txt
```

**Optional: Face Recognition Setup**

Face replacement works with OpenCV's built-in face detection by default (no additional installation needed). For better accuracy, you can install face_recognition:

**Easy Installation (Recommended):**

1. **Windows**: Run `install_face_recognition.bat` from the backend directory
2. **Linux/Mac**: Run `bash install_face_recognition.sh` from the backend directory

**Manual Installation:**

1. **Windows - Method 1 (Easiest - Pre-built wheel):**
```bash
cd backend
venv\Scripts\activate
pip install dlib-binary
pip install face-recognition
```

2. **Windows - Method 2 (Using conda):**
```bash
conda install -c conda-forge dlib
pip install face-recognition
```

3. **Windows - Method 3 (From source):**
```bash
pip install cmake
pip install dlib
pip install face-recognition
```

4. **Linux/Mac:**
```bash
pip install cmake
pip install dlib
pip install face-recognition
```

**Note**: Face replacement will work without face_recognition using OpenCV's Haar Cascade detector, but face_recognition provides better accuracy and detection.

**Note**: If you encounter setuptools errors, make sure to upgrade pip, setuptools, and wheel before installing other packages.

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

## Running the Application

### Start the Backend

1. Activate your virtual environment (if not already activated)
2. Navigate to the backend directory
3. Run the FastAPI server:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

### Start the Frontend

1. Navigate to the frontend directory
2. Start the React development server:
```bash
npm start
```

The frontend will be available at `http://localhost:3000`

## Usage

1. **Upload Input**: Choose to upload a photo/video OR enter a text prompt
2. **Set Duration**: Select 5 or 10 seconds for your video
3. **Face Replacement** (Optional): 
   - Enable "Replace face in video"
   - Upload your face image
4. **Generate**: Click "Generate Video" and wait for processing
5. **Download**: Preview and download your generated video

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

- **Video Resolution**: 1280x720 (720p)
- **Frame Rate**: 24 fps
- **Video Format**: MP4 (H.264)
- **Face Detection**: Uses face_recognition library
- **Video Processing**: OpenCV and FFmpeg

## Notes

- The current implementation uses basic video generation techniques. For production use, consider integrating:
  - Stable Video Diffusion for text-to-video
  - AnimateDiff for image animation
  - SimSwap or FaceSwap-GAN for advanced face replacement
- Face replacement requires clear face images for best results
- Processing time depends on video duration and system capabilities

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
If you encounter issues installing dlib, try:
- **Windows**: Use conda: `conda install -c conda-forge dlib`
- **Windows Alternative**: Use pre-built wheel: `pip install dlib-binary`
- **Linux/Mac**: Install CMake first: `pip install cmake` or `sudo apt-get install cmake`
- Face recognition will work without dlib, but with limited functionality

### FFmpeg Not Found
Install FFmpeg:
- **Windows**: Download from https://ffmpeg.org/download.html and add to PATH
- **Linux**: `sudo apt-get install ffmpeg`
- **Mac**: `brew install ffmpeg`

**Note**: FFmpeg is optional but recommended for better video encoding. The app will work without it but may produce larger files.

### Face Recognition Errors
If face recognition fails, ensure:
- Face images are clear and front-facing
- The shape predictor model is downloaded (optional but recommended)
- Sufficient lighting in images
- Face recognition libraries are properly installed (see Optional Setup above)

### Video Generation Works Without Face Recognition
The core video generation features (photo-to-video, text-to-video, video processing) work without face recognition. Face replacement is an optional feature that requires additional setup.

## License

This project is open source and available for modification and distribution.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.