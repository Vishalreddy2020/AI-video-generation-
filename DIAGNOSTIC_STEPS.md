# Diagnostic Steps - "Link Not Working"

## Quick Check

Run this first:
```bash
python test_backend.py
```

Or check manually:
```bash
curl http://localhost:8000/api/health
```

## Step-by-Step Diagnosis

### 1. Is Backend Running?

**Check:**
- Open terminal where you ran `python main.py`
- Should see: `INFO: Uvicorn running on http://0.0.0.0:8000`

**If not running:**
```bash
cd backend
venv\Scripts\activate
python main.py
```

### 2. Is Frontend Running?

**Check:**
- Open terminal where you ran `npm start`
- Should see: `Compiled successfully!` and `Local: http://localhost:3000`

**If not running:**
```bash
cd frontend
npm install
npm start
```

### 3. Test Backend Directly

Open browser and go to:
- http://localhost:8000/api/health

**Expected:** `{"status": "healthy"}`

**If error:**
- Backend is not running
- Port 8000 is blocked
- Backend crashed

### 4. Check Browser Console

1. Open http://localhost:3000
2. Press F12 (Developer Tools)
3. Go to Console tab
4. Look for errors

**Common errors:**

**"Network Error" or "ECONNREFUSED":**
```
Cannot connect to backend server at http://localhost:8000
```
→ Backend is not running. Start it with `start_backend.bat`

**CORS Error:**
```
Access to XMLHttpRequest blocked by CORS policy
```
→ Check CORS settings in `backend/main.py`

**404 Not Found:**
```
GET http://localhost:8000/api/health 404
```
→ Backend is running but route doesn't exist (unlikely)

### 5. Check Ports Are Available

**Windows:**
```bash
netstat -ano | findstr :8000
netstat -ano | findstr :3000
```

If ports are in use, kill the process or change ports.

### 6. Verify Installation

**Backend:**
```bash
cd backend
venv\Scripts\activate
pip list | findstr fastapi
```

**Frontend:**
```bash
cd frontend
npm list react
```

## Common Solutions

### Solution 1: Restart Everything

1. Stop both servers (Ctrl+C in each terminal)
2. Start backend: `start_backend.bat`
3. Start frontend: `start_frontend.bat`
4. Clear browser cache (Ctrl+Shift+Delete)

### Solution 2: Check Firewall

Windows Firewall might be blocking ports:
- Allow Python through firewall
- Allow Node.js through firewall

### Solution 3: Use Different Ports

If ports 3000/8000 are in use:

**Backend (change port 8000 to 8001):**
Edit `backend/main.py` line 121:
```python
uvicorn.run(app, host="0.0.0.0", port=8001)
```

**Frontend (change API URL):**
Create `frontend/.env`:
```
REACT_APP_API_URL=http://localhost:8001
```

### Solution 4: Check Virtual Environment

Make sure virtual environment is activated:
```bash
cd backend
venv\Scripts\activate
# Should see (venv) in prompt
python main.py
```

## Still Not Working?

1. **Check logs:**
   - Backend terminal output
   - Browser console (F12)
   - Network tab in browser DevTools

2. **Run diagnostic:**
   ```bash
   check_setup.bat
   ```

3. **Test backend manually:**
   ```bash
   python test_backend.py
   ```

4. **Check file structure:**
   - `backend/main.py` exists
   - `backend/services/` folder exists
   - `frontend/src/` folder exists

