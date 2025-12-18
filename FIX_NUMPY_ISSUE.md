# Fix NumPy Compatibility Issue

## Problem

You're seeing this error:
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.3.5
ImportError: numpy.core.multiarray failed to import
```

This happens because OpenCV was compiled with NumPy 1.x, but NumPy 2.x is installed.

## Quick Fix

**Option 1: Use the fix script (Easiest)**
```bash
fix_numpy.bat
```

**Option 2: Manual fix**
```bash
cd backend
venv\Scripts\activate
pip uninstall numpy -y
pip install "numpy<2.0.0"
```

## Verify Fix

After fixing, test:
```bash
python -c "import numpy; print('NumPy:', numpy.__version__)"
python -c "import cv2; print('OpenCV:', cv2.__version__)"
```

Both should work without errors.

## Then Start Backend

```bash
python main.py
```

The backend should start successfully now!

