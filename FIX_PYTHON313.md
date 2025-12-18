# Fix for Python 3.13 Compatibility

## Problem
Python 3.13 is very new and NumPy 1.x doesn't have pre-built wheels for it. Only NumPy 2.x is available.

## Solution: Upgrade OpenCV

Instead of downgrading NumPy, we'll upgrade OpenCV to a version that supports NumPy 2.x.

### Quick Fix Commands:

```bash
cd backend
venv\Scripts\activate
pip uninstall numpy opencv-python -y
pip install opencv-python-headless
pip install numpy
```

Or use the updated fix script:
```bash
fix_numpy.bat
```

### Why opencv-python-headless?

- `opencv-python-headless` is a newer version that supports NumPy 2.x
- It's the same as opencv-python but without GUI dependencies
- Perfect for server applications

### Verify Installation:

```bash
python -c "import numpy; import cv2; print('NumPy:', numpy.__version__); print('OpenCV:', cv2.__version__)"
```

Both should work now!

