#!/bin/bash
# CafeLocate Setup Script
# This script helps set up the CafeLocate project for development

echo "🍵 CafeLocate Setup Script"
echo "=========================="

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed. Please install Python 3.8+ first."
    exit 1
fi

echo "✅ Python found: $(python --version)"

# Check if we're in the right directory
if [ ! -d "cafelocate" ]; then
    echo "❌ Error: Please run this script from the MP2 directory (where cafelocate/ folder is located)"
    exit 1
fi

echo "📁 Setting up in directory: $(pwd)"

# Create virtual environment
echo "🔧 Creating virtual environment..."
python -m venv .venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate  # For Linux/Mac
# For Windows, this would be: .venv\Scripts\activate

# Install backend dependencies
echo "📦 Installing backend dependencies..."
cd cafelocate/backend
pip install -r requirements.txt

# Go back to project root
cd ../..

# Copy environment file
echo "📋 Setting up environment configuration..."
if [ ! -f "cafelocate/.env" ]; then
    cp cafelocate/.env.example cafelocate/.env
    echo "✅ Created .env file from template"
    echo "⚠️  Please edit cafelocate/.env and add your Mapbox API token!"
else
    echo "ℹ️  .env file already exists"
fi

# Run migrations
echo "🗄️  Setting up database..."
cd cafelocate/backend
python manage.py migrate

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit cafelocate/.env and add your Mapbox API token"
echo "2. Run: cd cafelocate/backend && python manage.py runserver"
echo "3. In another terminal: cd cafelocate/frontend && python -m http.server 5500"
echo "4. Open http://localhost:5500 in your browser"
echo ""
echo "For detailed instructions, see README.md"