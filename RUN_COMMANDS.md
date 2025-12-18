# Commands to Run the Application

## Windows Commands

### Quick Start (Easiest Method)

**Terminal 1 - Backend:**
```cmd
start_backend.bat
```

**Terminal 2 - Frontend:**
```cmd
start_frontend.bat
```

### Manual Commands

#### Backend (Terminal 1):
```cmd
cd backend
venv\Scripts\activate
python main.py
```

#### Frontend (Terminal 2 - New Window):
```cmd
cd frontend
npm install
npm start
```

## Linux/Mac Commands

### Quick Start

**Terminal 1 - Backend:**
```bash
cd backend
source venv/bin/activate
python main.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm install
npm start
```

## First Time Setup Commands

### Windows:
```cmd
cd backend
setup.bat
```

### Linux/Mac:
```bash
cd backend
bash setup.sh
```

## Complete Setup and Run (Windows)

```cmd
REM Step 1: Setup backend
cd backend
setup.bat

REM Step 2: Start backend (in first terminal)
cd backend
venv\Scripts\activate
python main.py

REM Step 3: Start frontend (in second terminal)
cd frontend
npm install
npm start
```

## Complete Setup and Run (Linux/Mac)

```bash
# Step 1: Setup backend
cd backend
bash setup.sh

# Step 2: Start backend (in first terminal)
cd backend
source venv/bin/activate
python main.py

# Step 3: Start frontend (in second terminal)
cd frontend
npm install
npm start
```

## URLs

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **Health Check**: http://localhost:8000/api/health

## Optional: Install Face Recognition

**Windows:**
```cmd
cd backend
install_face_recognition.bat
```

**Linux/Mac:**
```bash
cd backend
bash install_face_recognition.sh
```

