#!/bin/bash
echo "Starting AI Video Generator Frontend..."
cd frontend
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi
echo "Starting development server..."
npm start

