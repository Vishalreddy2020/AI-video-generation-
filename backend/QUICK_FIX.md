# Quick Fix for Installation Errors

## If you see "Cannot import 'setuptools.build_meta'" error:

1. **Activate your virtual environment:**
   ```bash
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

2. **Upgrade pip, setuptools, and wheel:**
   ```bash
   python -m pip install --upgrade pip setuptools wheel
   ```

3. **Install core dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Face Replacement Feature

**Good News**: Face replacement works immediately using OpenCV's built-in face detection (no installation needed)!

**For Better Accuracy**: Install face_recognition library for improved face detection:

**Easiest Method**: Run the installation script:
- Windows: `install_face_recognition.bat`
- Linux/Mac: `bash install_face_recognition.sh`

**Manual Installation Options:**

**Option 1: Pre-built wheel (Windows - Easiest)**
```bash
pip install dlib-binary
pip install face-recognition
```

**Option 2: Using conda (Windows recommended)**
```bash
conda install -c conda-forge dlib
pip install face-recognition
```

**Option 3: From source**
```bash
pip install cmake
pip install dlib
pip install face-recognition
```

**Note**: The app works fine without face_recognition - it uses OpenCV's Haar Cascade detector as a fallback. Face replacement will still work, just with slightly lower accuracy.

## If you still have issues:

1. Delete the `venv` folder
2. Run `setup.bat` (Windows) or `setup.sh` (Linux/Mac) again
3. This will create a fresh virtual environment with all dependencies

