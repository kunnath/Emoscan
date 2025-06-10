#!/bin/bash

# EmoScan Launcher Script for macOS/Linux
# This script helps launch the EmoScan application with proper setup

echo "🎭 EmoScan - Emotion & Body Language Analyzer"
echo "=============================================="

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "emoscan_env" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run the setup first:"
    echo "  python3 -m venv emoscan_env"
    echo "  source emoscan_env/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source emoscan_env/bin/activate

# Check if required packages are installed
echo "🔍 Checking installation..."
python test_installation.py || {
    echo "❌ Installation check failed!"
    echo "Please run: pip install -r requirements.txt"
    exit 1
}

echo "🚀 Starting EmoScan application..."
streamlit run app.py
    echo "❌ Python 3 is required but not installed."
    echo "Please install Python 3.8+ from https://python.org"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "❌ app.py not found. Please run this script from the EmoScan directory."
    exit 1
fi

# Check if virtual environment exists
VENV_PATH="emoscan_env"
if [ ! -d "$VENV_PATH" ]; then
    echo "⚠️  Virtual environment not found at $VENV_PATH"
    echo "🔧 Setting up virtual environment..."
    
    if [ -f "setup_venv.py" ]; then
        python3 setup_venv.py
        if [ $? -ne 0 ]; then
            echo "❌ Virtual environment setup failed!"
            exit 1
        fi
    else
        echo "📦 Creating virtual environment manually..."
        python3 -m venv "$VENV_PATH"
        echo "📦 Installing requirements..."
        source "$VENV_PATH/bin/activate"
        pip install --upgrade pip
        pip install -r requirements.txt
    fi
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source "$VENV_PATH/bin/activate"

if [ $? -ne 0 ]; then
    echo "❌ Failed to activate virtual environment!"
    exit 1
fi

echo "✅ Virtual environment activated"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/upgrade pip
pip install --upgrade pip

# Install requirements
if [ -f "requirements.txt" ]; then
    echo "📦 Installing requirements..."
    pip install -r requirements.txt
else
    echo "⚠️  requirements.txt not found. Installing basic packages..."
    pip install streamlit opencv-python numpy pandas
fi

# Check if Streamlit is available
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit installation failed."
    exit 1
fi

echo ""
echo "✅ Setup complete!"
echo ""

# Ask user what to do
echo "What would you like to do?"
echo "1) Run setup and diagnostics (recommended for first time)"
echo "2) Run quick demo"
echo "3) Start the main application"
echo "4) Exit"
echo ""

read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo "🔧 Running setup and diagnostics..."
        python3 setup.py
        ;;
    2)
        echo "🎬 Running quick demo..."
        python3 demo.py
        ;;
    3)
        echo "🚀 Starting EmoScan application..."
        echo "The app will open in your browser at http://localhost:8501"
        echo "Press Ctrl+C to stop the application"
        echo ""
        streamlit run app.py
        ;;
    4)
        echo "👋 Goodbye!"
        exit 0
        ;;
    *)
        echo "❌ Invalid choice. Starting main application by default..."
        streamlit run app.py
        ;;
esac
