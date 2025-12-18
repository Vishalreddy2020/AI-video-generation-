# Face Replacement Feature Guide

## Overview

The AI Video Generator includes a face replacement feature that allows you to replace faces in generated videos with your own face image.

## How It Works

1. **Generate a video** from a photo, text prompt, or existing video
2. **Enable "Replace face in video"** option
3. **Upload your face image** (clear, front-facing photo works best)
4. The system will detect faces in the generated video and replace them with your face

## Face Detection Methods

The system uses multiple face detection methods with automatic fallback:

1. **face_recognition library** (if installed) - Most accurate
2. **OpenCV DNN** (if models available) - Good accuracy
3. **OpenCV Haar Cascade** (built-in) - Works immediately, no installation needed

## Installation

### Quick Start (Works Immediately)

Face replacement works **right away** using OpenCV's built-in Haar Cascade detector. No installation needed!

### Enhanced Accuracy (Optional)

For better face detection accuracy, install the face_recognition library:

**Windows:**
```bash
cd backend
.\install_face_recognition.bat
```

**Linux/Mac:**
```bash
cd backend
bash install_face_recognition.sh
```

Or manually:
```bash
# Windows (easiest)
pip install dlib-binary
pip install face-recognition

# Or using conda
conda install -c conda-forge dlib
pip install face-recognition
```

## Usage Tips

1. **Face Image Quality:**
   - Use a clear, front-facing photo
   - Good lighting
   - Face should be clearly visible
   - Avoid sunglasses or face coverings

2. **Video Quality:**
   - Videos with clear, front-facing faces work best
   - Multiple faces: The first detected face will be replaced
   - Side profiles may not be detected as well

3. **Best Results:**
   - Use face_recognition library for better accuracy
   - Ensure good lighting in both source video and face image
   - Face sizes should be similar for best blending

## Troubleshooting

### "Could not detect face in target image"
- Use a clearer, front-facing photo
- Ensure good lighting
- Make sure the face is clearly visible

### Face not being replaced in video
- The video may not have detectable faces
- Try installing face_recognition for better detection
- Ensure faces in video are front-facing and well-lit

### Installation Issues
- See `QUICK_FIX.md` for troubleshooting
- Face replacement works without face_recognition (using OpenCV fallback)
- You can always use the basic face detection that comes with OpenCV

## Technical Details

- **Blending Method**: Uses Gaussian blur mask for smooth face blending
- **Face Detection**: Automatic fallback between multiple detection methods
- **Video Processing**: Maintains original video quality and frame rate
- **Output Format**: MP4 (H.264) at 720p resolution

