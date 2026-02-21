@echo off
REM CafeLocate Setup Script for Windows
REM This script helps set up the CafeLocate project for development

echo 🍵 CafeLocate Setup Script
echo ==========================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.8+ first.
    pause
    exit /b 1
)

echo ✅ Python found:
python --version

REM Check if we're in the right directory
if not exist "cafelocate" (
    echo ❌ Error: Please run this script from the MP2 directory (where cafelocate\ folder is located)
    pause
    exit /b 1
)

echo 📁 Setting up in directory: %cd%

REM Create virtual environment
echo 🔧 Creating virtual environment...
python -m venv .venv

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call .venv\Scripts\activate

REM Install backend dependencies
echo 📦 Installing backend dependencies...
cd cafelocate\backend
pip install -r requirements.txt

REM Go back to project root
cd ..\..

REM Copy environment file
echo 📋 Setting up environment configuration...
if not exist "cafelocate\.env" (
    copy cafelocate\.env.example cafelocate\.env
    echo ✅ Created .env file from template
    echo ⚠️  Please edit cafelocate\.env and add your Mapbox API token!
) else (
    echo ℹ️  .env file already exists
)

REM Run migrations
echo 🗄️  Setting up database...
cd cafelocate\backend
python manage.py migrate

echo.
echo 🎉 Setup complete!
echo.
echo Next steps:
echo 1. Edit cafelocate\.env and add your Mapbox API token
echo 2. Run: cd cafelocate\backend ^&^& python manage.py runserver
echo 3. In another terminal: cd cafelocate\frontend ^&^& python -m http.server 5500
echo 4. Open http://localhost:5500 in your browser
echo.
echo For detailed instructions, see README.md

pause