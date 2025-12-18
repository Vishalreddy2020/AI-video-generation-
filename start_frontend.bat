@echo off
echo Starting AI Video Generator Frontend...
cd frontend
if not exist node_modules (
    echo Installing dependencies...
    call npm install
)
echo Starting development server...
call npm start

