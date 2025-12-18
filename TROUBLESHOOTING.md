# Troubleshooting Guide

## "The link is not working" / Connection Issues

### Step 1: Check if Backend is Running

**Test the backend connection:**
```bash
python test_backend.py
```

Or manually test:
```bash
curl http://localhost:8000/api/health
```

**Expected response:**
```json
{"status": "healthy"}
```

### Step 2: Check if Frontend is Running

Open browser console (F12) and check for errors.

**Common errors:**

1. **"Network Error" or "ECONNREFUSED"**
   - Backend is not running
   - Solution: Start backend with `start_backend.bat` or `python backend/main.py`

2. **"CORS policy" error**
   - Backend CORS settings issue
   - Solution: Check `backend/main.py` CORS configuration

3. **"Cannot GET /"**
   - Frontend not running
   - Solution: Start frontend with `start_frontend.bat` or `npm start`

### Step 3: Verify Both Servers Are Running

**Backend should show:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Frontend should show:**
```
Compiled successfully!
You can now view ai-video-generator-frontend in the browser.
  Local:            http://localhost:3000
```

### Step 4: Check Ports

**Backend Port (8000):**
```bash
netstat -ano | findstr :8000
```

**Frontend Port (3000):**
```bash
netstat -ano | findstr :3000
```

If ports are in use:
- Backend: Change port in `backend/main.py` line 121
- Frontend: React will auto-use next available port

### Step 5: Check Browser Console

1. Open browser (Chrome/Firefox)
2. Press F12 to open Developer Tools
3. Go to Console tab
4. Look for red error messages
5. Check Network tab for failed requests

### Common Issues and Solutions

#### Issue: "Cannot connect to backend server"

**Solution:**
1. Make sure backend is running:
   ```bash
   cd backend
   venv\Scripts\activate
   python main.py
   ```

2. Check if backend started successfully (should see "Uvicorn running")

3. Test backend directly:
   ```bash
   curl http://localhost:8000/api/health
   ```

#### Issue: Frontend shows error but backend seems running

**Solution:**
1. Check CORS settings in `backend/main.py`
2. Make sure frontend URL is in allowed origins
3. Restart both servers

#### Issue: "Module not found" errors

**Solution:**
1. Backend: Run `setup.bat` again
2. Frontend: Run `npm install` in frontend directory

#### Issue: Port already in use

**Solution:**
1. Find process using port:
   ```bash
   netstat -ano | findstr :8000
   ```
2. Kill the process or change port in code

### Quick Diagnostic Checklist

- [ ] Backend server is running (`python backend/main.py`)
- [ ] Frontend server is running (`npm start` in frontend/)
- [ ] Backend accessible at http://localhost:8000/api/health
- [ ] Frontend accessible at http://localhost:3000
- [ ] No firewall blocking ports 3000 and 8000
- [ ] Browser console shows no CORS errors
- [ ] Virtual environment is activated for backend

### Still Not Working?

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
   curl -X POST http://localhost:8000/api/generate-video \
     -F "text_prompt=test" \
     -F "duration=5"
   ```

