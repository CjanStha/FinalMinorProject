@echo off
REM CafeLocate Run Script for Windows
REM Starts both backend and frontend servers

echo 🍵 Starting CafeLocate Servers
echo ==============================

REM Check if virtual environment exists
if not exist ".venv" (
    echo ❌ Virtual environment not found. Please run setup.bat first.
    pause
    exit /b 1
)

REM Check if .env file exists
if not exist "cafelocate\.env" (
    echo ❌ .env file not found. Please run setup.bat first.
    pause
    exit /b 1
)

echo 🔧 Activating virtual environment...
call .venv\Scripts\activate

echo 🚀 Starting Django backend server...
start "CafeLocate Backend" cmd /k "cd cafelocate\backend && python manage.py runserver"

timeout /t 3 /nobreak >nul

echo 🌐 Starting frontend server...
start "CafeLocate Frontend" cmd /k "cd cafelocate\frontend && python -m http.server 5500"

echo.
echo 🎉 Servers started!
echo.
echo Backend API: http://localhost:8000
echo Frontend App: http://localhost:5500
echo.
echo Press any key to close this window...
pause >nul