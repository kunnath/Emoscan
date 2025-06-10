@echo off
REM EmoScan Launcher Script for Windows
REM This script helps launch the EmoScan application with proper setup

echo 🎭 EmoScan - Emotion ^& Body Language Analyzer
echo ==============================================

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is required but not installed.
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

REM Check if we're in the right directory
if not exist "app.py" (
    echo ❌ app.py not found. Please run this script from the EmoScan directory.
    pause
    exit /b 1
)

REM Check if virtual environment exists
set VENV_PATH=emoscan_env
if not exist "%VENV_PATH%" (
    echo ⚠️  Virtual environment not found at %VENV_PATH%
    echo 🔧 Setting up virtual environment...
    
    if exist "setup_venv.py" (
        python setup_venv.py
        if %errorlevel% neq 0 (
            echo ❌ Virtual environment setup failed!
            pause
            exit /b 1
        )
    ) else (
        echo 📦 Creating virtual environment manually...
        python -m venv "%VENV_PATH%"
        echo 📦 Installing requirements...
        call "%VENV_PATH%\Scripts\activate.bat"
        pip install --upgrade pip
        pip install -r requirements.txt
    )
)

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call "%VENV_PATH%\Scripts\activate.bat"

if %errorlevel% neq 0 (
    echo ❌ Failed to activate virtual environment!
    pause
    exit /b 1
)

echo ✅ Virtual environment activated
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install/upgrade pip
pip install --upgrade pip

REM Install requirements
if exist "requirements.txt" (
    echo 📦 Installing requirements...
    pip install -r requirements.txt
) else (
    echo ⚠️  requirements.txt not found. Installing basic packages...
    pip install streamlit opencv-python numpy pandas
)

echo.
echo ✅ Setup complete!
echo.

REM Ask user what to do
echo What would you like to do?
echo 1) Run setup and diagnostics (recommended for first time)
echo 2) Run quick demo
echo 3) Start the main application
echo 4) Exit
echo.

set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    echo 🔧 Running setup and diagnostics...
    python setup.py
) else if "%choice%"=="2" (
    echo 🎬 Running quick demo...
    python demo.py
) else if "%choice%"=="3" (
    echo 🚀 Starting EmoScan application...
    echo The app will open in your browser at http://localhost:8501
    echo Press Ctrl+C to stop the application
    echo.
    streamlit run app.py
) else if "%choice%"=="4" (
    echo 👋 Goodbye!
    exit /b 0
) else (
    echo ❌ Invalid choice. Starting main application by default...
    streamlit run app.py
)

pause
