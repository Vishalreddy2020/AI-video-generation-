# Troubleshooting Guide

## Connection Issues

### Backend Not Responding

**Test the backend connection:**
```bash
curl http://localhost:8000/api/health
```

**Expected response:**
```json
{"status": "healthy"}
```

**If connection fails:**
1. Verify backend is running: Check terminal where `python main.py` is running
2. Check for error messages in backend terminal
3. Verify port 8000 is not in use: `netstat -ano | findstr :8000` (Windows) or `lsof -i :8000` (Linux/Mac)
4. Restart backend server

### Frontend Cannot Connect to Backend

**Symptoms:**
- Frontend loads but shows "Cannot connect to backend" error
- Network errors in browser console (F12)

**Solutions:**
1. Verify backend is running and accessible at http://localhost:8000/api/health
2. Check CORS settings in `backend/main.py` - ensure frontend URL is in allowed origins
3. Check browser console (F12) for specific error messages
4. Verify both servers are running on correct ports
5. Check firewall settings - ports 3000 and 8000 should be accessible

### Port Already in Use

**Backend (port 8000):**
- Find process using port: `netstat -ano | findstr :8000` (Windows)
- Kill the process or change port in `backend/main.py` line 121

**Frontend (port 3000):**
- React will automatically use next available port (3001, 3002, etc.)
- Or manually change in `frontend/package.json`

## Installation Issues

### Setuptools/Build Errors

**Error:** "Cannot import 'setuptools.build_meta'" or "BackendUnavailable"

**Solution:**
```bash
cd backend
venv\Scripts\activate  # Windows
# or: source venv/bin/activate  # Linux/Mac
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Module Not Found Errors

**Backend:**
```bash
cd backend
venv\Scripts\activate
pip install -r requirements.txt
```

**Frontend:**
```bash
cd frontend
npm install
```

### dlib Installation Issues

**Windows - Method 1 (Easiest):**
```bash
pip install dlib-binary
pip install face-recognition
```

**Windows - Method 2 (Using conda):**
```bash
conda install -c conda-forge dlib
pip install face-recognition
```

**Linux/Mac:**
```bash
pip install cmake
pip install dlib
pip install face-recognition
```

**Note:** Face replacement works without dlib using OpenCV's Haar Cascade detector.

### NumPy/OpenCV Compatibility Issues

**Python 3.13+ users:**
```bash
pip install opencv-python-headless>=4.9.0
pip install numpy>=1.24.0
```

## Video Generation Issues

### Video Generation Fails

**Check:**
1. Backend logs for error messages
2. Sufficient disk space (videos are saved to `backend/outputs/`)
3. FFmpeg is installed (optional but recommended)
4. Input file format is supported (jpg, png, mp4, etc.)

### Face Replacement Not Working

**If face detection fails:**
1. Ensure face is clearly visible and front-facing
2. Try installing face_recognition for better accuracy (see Installation Issues)
3. Check that face image is in supported format (jpg, png)
4. Verify sufficient lighting in images

**Note:** Face replacement works without face_recognition using OpenCV's built-in detection, but with lower accuracy.

### Out of Memory Errors

**Solutions:**
1. Close other applications to free up RAM
2. Reduce video duration (use 5 seconds instead of 10)
3. Use smaller input images
4. If using GPU, reduce batch size in code

## Performance Issues

### Slow Video Generation

**CPU Mode:**
- Normal: 5-15 minutes per 5-second video
- Consider using GPU for faster processing

**GPU Mode:**
- Should be 1-3 minutes per 5-second video
- If slower, check GPU drivers are up to date

### First Generation Takes Long Time

**Normal behavior:**
- Models download on first use (~4-8GB)
- Subsequent generations will be faster
- Models are cached for future use

## Quick Diagnostic Checklist

- [ ] Backend server is running (`python backend/main.py`)
- [ ] Frontend server is running (`npm start` in frontend/)
- [ ] Backend accessible at http://localhost:8000/api/health
- [ ] Frontend accessible at http://localhost:3000
- [ ] Virtual environment is activated for backend
- [ ] All dependencies installed (check with `pip list` and `npm list`)
- [ ] No firewall blocking ports 3000 and 8000
- [ ] Browser console shows no CORS errors
- [ ] Sufficient disk space available

## Still Not Working?

1. **Check logs:**
   - Backend: Look at terminal where `python main.py` is running
   - Frontend: Check browser console (F12)

2. **Restart everything:**
   - Stop both servers (Ctrl+C)
   - Restart backend
   - Restart frontend
   - Clear browser cache (Ctrl+Shift+Delete)

3. **Verify installation:**
   ```bash
   # Backend
   cd backend
   venv\Scripts\activate
   pip list
   
   # Frontend
   cd frontend
   npm list
   ```

4. **Test with curl/Postman:**
   ```bash
   curl http://localhost:8000/api/health
   ```

5. **Recreate virtual environment:**
   ```bash
   cd backend
   rm -rf venv  # or rmdir /s venv on Windows
   setup.bat  # or bash setup.sh
   ```
