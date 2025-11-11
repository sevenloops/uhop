#!/bin/bash
# Start the UHOP demo portal

echo "Starting UHOP Demo Portal..."

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    python -m venv .venv
fi

# Activate virtual environment
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    source .venv/Scripts/activate
else
    source .venv/bin/activate
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -e .

# Install Node.js dependencies
echo "Installing Node.js dependencies..."
cd frontend
npm install
cd ..

cd backend
npm install
cd ..

# Start backend in background
echo "Starting backend server..."
cd backend
node index.js &
BACKEND_PID=$!
cd ..

# Start frontend
echo "Starting frontend server..."
cd frontend
npm run dev &
FRONTEND_PID=$!

echo ""
echo "UHOP Demo Portal is starting..."
echo "Frontend: http://localhost:5173"
echo "Backend: http://localhost:8787"
echo ""
echo "Press Ctrl+C to stop both servers"

# Wait for interrupt
wait

# Cleanup
echo "Stopping servers..."
kill $BACKEND_PID 2>/dev/null
kill $FRONTEND_PID 2>/dev/null
