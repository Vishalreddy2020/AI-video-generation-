# Quick Start Guide - AI Video Generator

## Prerequisites Check

Make sure you have:
- ✅ Python 3.8+ installed
- ✅ Node.js 16+ installed
- ✅ Virtual environment set up (run `backend\setup.bat` if not done)

## Running the Application

### Option 1: Using Batch Scripts (Windows - Easiest)

**Terminal 1 - Backend:**
```bash
start_backend.bat
```

**Terminal 2 - Frontend:**
```bash
start_frontend.bat
```

### Option 2: Manual Commands

#### Step 1: Start Backend Server

Open a terminal/command prompt:

```bash
cd backend
venv\Scripts\activate
python main.py
```

The backend will start at: **http://localhost:8000**

#### Step 2: Start Frontend (New Terminal)

Open a **new** terminal/command prompt:

```bash
cd frontend
npm install
npm start
```

The frontend will start at: **http://localhost:3000**

## First Time Setup

If you haven't set up the project yet:

### Backend Setup:
```bash
cd backend
setup.bat
```

This will:
- Create virtual environment
- Install all dependencies
- Set up the project

### Frontend Setup:
```bash
cd frontend
npm install
```

## Access the Application

Once both servers are running:
- **Frontend UI**: Open your browser and go to **http://localhost:3000**
- **Backend API**: Available at **http://localhost:8000**
- **API Health Check**: http://localhost:8000/api/health

## Troubleshooting

### Backend won't start
- Make sure virtual environment is activated: `venv\Scripts\activate`
- Check if port 8000 is available
- Run `setup.bat` again if dependencies are missing

### Frontend won't start
- Make sure Node.js is installed: `node --version`
- Run `npm install` in the frontend directory
- Check if port 3000 is available

### Port already in use
- Backend: Change port in `backend/main.py` (line 121)
- Frontend: React will automatically use the next available port

## Stopping the Servers

- **Backend**: Press `Ctrl+C` in the backend terminal
- **Frontend**: Press `Ctrl+C` in the frontend terminal

## Optional: Install Face Recognition (Better Face Detection)

For enhanced face replacement accuracy:
```bash
cd backend
install_face_recognition.bat
```

Note: Face replacement works without this, but with slightly lower accuracy.

