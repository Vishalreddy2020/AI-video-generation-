# Fix NumPy Installation Issue

## Problem
NumPy is trying to build from source but no C compiler is found. Error:
```
ERROR: Unknown compiler(s): [['icl'], ['cl'], ['cc'], ['gcc'], ['clang'], ['clang-cl'], ['pgcc']]
```

## Solution: Use Pre-built Wheel

Install NumPy using a pre-built wheel (no compilation needed):

```bash
cd backend
venv\Scripts\activate
pip uninstall numpy -y
pip install --only-binary :all: "numpy<2.0.0"
```

Or specify a specific version with pre-built wheels:
```bash
pip install --only-binary :all: "numpy==1.26.4"
```

## Alternative: Use pip with --prefer-binary

```bash
pip install --prefer-binary "numpy<2.0.0"
```

## Verify Installation

```bash
python -c "import numpy; import cv2; print('NumPy:', numpy.__version__); print('OpenCV:', cv2.__version__)"
```

Both should import successfully!

