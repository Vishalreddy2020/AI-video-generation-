#!/bin/bash
echo "Installing Face Recognition Dependencies..."
echo
echo "This script will install face_recognition and dlib."
echo

cd backend
if [ ! -d "venv" ]; then
    echo "ERROR: Virtual environment not found. Please run setup.sh first!"
    exit 1
fi

source venv/bin/activate

echo
echo "Installing CMake (required for dlib)..."
pip install cmake

echo
echo "Installing dlib..."
pip install dlib

echo
echo "Installing face-recognition..."
pip install face-recognition

echo
echo "========================================"
echo "Installation completed!"
echo "========================================"
echo
echo "Face replacement feature should now work."
echo "Restart the backend server to use it."
echo

