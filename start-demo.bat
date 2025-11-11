@echo off
REM Start the UHOP demo portal on Windows

echo Starting UHOP Demo Portal...

REM Check if virtual environment exists
if not exist ".venv" (
    echo Creating Python virtual environment...
    python -m venv .venv
)

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Install Python dependencies
echo Installing Python dependencies...
pip install -e .

REM Install Node.js dependencies
echo Installing Node.js dependencies...
cd frontend
call npm install
cd ..

cd backend
call npm install
cd ..

REM Start backend
echo Starting backend server...
cd backend
start /b node index.js
cd ..

REM Start frontend
echo Starting frontend server...
cd frontend
start /b npm run dev

echo.
echo UHOP Demo Portal is starting...
echo Frontend: http://localhost:5173
echo Backend: http://localhost:8787
echo.
echo Press any key to stop the demo
pause > nul

REM Note: On Windows, you might need to manually stop the processes
echo Please manually close the terminal windows to stop the servers.
