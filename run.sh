#!/bin/bash
# CafeLocate Run Script
# Starts both backend and frontend servers

echo "🍵 Starting CafeLocate Servers"
echo "=============================="

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Please run setup.sh first."
    exit 1
fi

# Check if .env file exists
if [ ! -f "cafelocate/.env" ]; then
    echo "❌ .env file not found. Please run setup.sh first."
    exit 1
fi

echo "🔧 Activating virtual environment..."
source .venv/bin/activate

echo "🚀 Starting Django backend server..."
cd cafelocate/backend
python manage.py runserver &
BACKEND_PID=$!

echo "🌐 Starting frontend server..."
cd ../frontend
python -m http.server 5500 &
FRONTEND_PID=$!

echo ""
echo "🎉 Servers started!"
echo ""
echo "Backend API: http://localhost:8000"
echo "Frontend App: http://localhost:5500"
echo ""
echo "Press Ctrl+C to stop all servers"

# Wait for user interrupt
trap "echo 'Stopping servers...'; kill $BACKEND_PID $FRONTEND_PID; exit" INT
wait