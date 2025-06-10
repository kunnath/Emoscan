#!/usr/bin/env python3
"""
Virtual Environment Setup Script for EmoScan
Creates and configures a Python virtual environment with all dependencies
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

class VenvSetup:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.venv_name = "emoscan_env"
        self.venv_path = self.project_root / self.venv_name
        self.requirements_file = self.project_root / "requirements.txt"
        
    def print_header(self):
        """Print setup header"""
        print("=" * 60)
        print("🎭 EmoScan Virtual Environment Setup")
        print("=" * 60)
        print(f"Project: {self.project_root}")
        print(f"Virtual Environment: {self.venv_path}")
        print("=" * 60)
    
    def check_python_version(self):
        """Check Python version compatibility"""
        min_version = (3, 8)
        current_version = sys.version_info[:2]
        
        print(f"🐍 Checking Python version...")
        print(f"   Current: {sys.version}")
        
        if current_version < min_version:
            print(f"❌ Python {min_version[0]}.{min_version[1]}+ required")
            return False
        
        print("✅ Python version compatible")
        return True
    
    def check_venv_module(self):
        """Check if venv module is available"""
        try:
            import venv
            print("✅ venv module available")
            return True
        except ImportError:
            print("❌ venv module not found")
            print("   Please install python3-venv package")
            return False
    
    def create_virtual_environment(self):
        """Create the virtual environment"""
        if self.venv_path.exists():
            print(f"📁 Virtual environment already exists at {self.venv_path}")
            response = input("   Do you want to recreate it? (y/N): ").lower()
            if response == 'y':
                print("🗑️  Removing existing virtual environment...")
                import shutil
                shutil.rmtree(self.venv_path)
            else:
                return True
        
        print(f"🔨 Creating virtual environment...")
        try:
            subprocess.run([
                sys.executable, "-m", "venv", str(self.venv_path)
            ], check=True)
            print("✅ Virtual environment created successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create virtual environment: {e}")
            return False
    
    def get_activation_script(self):
        """Get the appropriate activation script path"""
        if platform.system() == "Windows":
            return self.venv_path / "Scripts" / "activate"
        else:
            return self.venv_path / "bin" / "activate"
    
    def get_python_executable(self):
        """Get the Python executable in the virtual environment"""
        if platform.system() == "Windows":
            return self.venv_path / "Scripts" / "python.exe"
        else:
            return self.venv_path / "bin" / "python"
    
    def get_pip_executable(self):
        """Get the pip executable in the virtual environment"""
        if platform.system() == "Windows":
            return self.venv_path / "Scripts" / "pip.exe"
        else:
            return self.venv_path / "bin" / "pip"
    
    def upgrade_pip(self):
        """Upgrade pip in the virtual environment"""
        print("📦 Upgrading pip...")
        try:
            subprocess.run([
                str(self.get_python_executable()), "-m", "pip", "install", "--upgrade", "pip"
            ], check=True)
            print("✅ pip upgraded successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to upgrade pip: {e}")
            return False
    
    def install_requirements(self):
        """Install packages from requirements.txt"""
        if not self.requirements_file.exists():
            print(f"❌ Requirements file not found: {self.requirements_file}")
            return False
        
        print("📦 Installing requirements...")
        try:
            subprocess.run([
                str(self.get_pip_executable()), "install", "-r", str(self.requirements_file)
            ], check=True)
            print("✅ Requirements installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install requirements: {e}")
            return False
    
    def create_activation_scripts(self):
        """Create easy activation scripts"""
        # Create activation script for Unix/macOS
        activate_script = self.project_root / "activate_env.sh"
        with open(activate_script, 'w') as f:
            f.write(f"""#!/bin/bash
# EmoScan Environment Activation Script
echo "🎭 Activating EmoScan environment..."
source {self.venv_path}/bin/activate
echo "✅ Environment activated!"
echo "📍 To deactivate, run: deactivate"
echo ""
echo "🚀 To run the app:"
echo "   streamlit run app.py"
echo ""
# Keep shell open
exec "$SHELL"
""")
        
        # Make executable
        os.chmod(activate_script, 0o755)
        
        # Create Windows batch file
        activate_bat = self.project_root / "activate_env.bat"
        with open(activate_bat, 'w') as f:
            f.write(f"""@echo off
REM EmoScan Environment Activation Script
echo 🎭 Activating EmoScan environment...
call {self.venv_path}\\Scripts\\activate.bat
echo ✅ Environment activated!
echo 📍 To deactivate, run: deactivate
echo.
echo 🚀 To run the app:
echo    streamlit run app.py
echo.
cmd /k
""")
        
        # Create Python activation script
        activate_py = self.project_root / "activate_env.py"
        with open(activate_py, 'w') as f:
            f.write(f"""#!/usr/bin/env python3
'''
EmoScan Environment Activation Script
Use this to activate the virtual environment from Python
'''

import os
import sys
import subprocess
from pathlib import Path

def activate_environment():
    project_root = Path(__file__).parent
    venv_path = project_root / "{self.venv_name}"
    
    if not venv_path.exists():
        print("❌ Virtual environment not found!")
        print("   Run: python setup_venv.py")
        return False
    
    # Set environment variables
    os.environ['VIRTUAL_ENV'] = str(venv_path)
    os.environ['PATH'] = str(venv_path / 'bin') + os.pathsep + os.environ.get('PATH', '')
    
    # Update sys.path to use venv packages
    import site
    site.addsitedir(str(venv_path / 'lib' / 'python{{}}.{{}}'.format(*sys.version_info[:2]) / 'site-packages'))
    
    print("✅ Environment activated in Python!")
    return True

if __name__ == "__main__":
    activate_environment()
""")
        
        print("✅ Activation scripts created:")
        print(f"   📄 {activate_script}")
        print(f"   📄 {activate_bat}")
        print(f"   📄 {activate_py}")
    
    def create_run_scripts(self):
        """Create scripts to run the application"""
        # Unix/macOS run script
        run_script = self.project_root / "run_app.sh"
        with open(run_script, 'w') as f:
            f.write(f"""#!/bin/bash
# EmoScan Application Runner
echo "🎭 Starting EmoScan application..."

# Check if virtual environment exists
if [ ! -d "{self.venv_path}" ]; then
    echo "❌ Virtual environment not found!"
    echo "   Run: python3 setup_venv.py"
    exit 1
fi

# Activate environment and run app
source {self.venv_path}/bin/activate
echo "✅ Environment activated"
echo "🚀 Starting Streamlit app..."
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
""")
        
        os.chmod(run_script, 0o755)
        
        # Windows run script
        run_bat = self.project_root / "run_app.bat"
        with open(run_bat, 'w') as f:
            f.write(f"""@echo off
REM EmoScan Application Runner
echo 🎭 Starting EmoScan application...

REM Check if virtual environment exists
if not exist "{self.venv_path}" (
    echo ❌ Virtual environment not found!
    echo    Run: python setup_venv.py
    pause
    exit /b 1
)

REM Activate environment and run app
call {self.venv_path}\\Scripts\\activate.bat
echo ✅ Environment activated
echo 🚀 Starting Streamlit app...
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
pause
""")
        
        print("✅ Run scripts created:")
        print(f"   📄 {run_script}")
        print(f"   📄 {run_bat}")
    
    def print_usage_instructions(self):
        """Print usage instructions"""
        print("\n" + "=" * 60)
        print("🎉 Virtual Environment Setup Complete!")
        print("=" * 60)
        print("\n📋 Usage Instructions:")
        print("\n1. Activate the environment:")
        
        if platform.system() == "Windows":
            print("   • Windows: activate_env.bat")
            print("   • Or: .\\emoscan_env\\Scripts\\activate")
        else:
            print("   • Unix/macOS: ./activate_env.sh")
            print("   • Or: source emoscan_env/bin/activate")
        
        print("\n2. Run the application:")
        print("   • Easy way: ./run_app.sh (Unix/macOS) or run_app.bat (Windows)")
        print("   • Manual: streamlit run app.py")
        
        print("\n3. Deactivate when done:")
        print("   • Run: deactivate")
        
        print("\n📁 Virtual Environment Location:")
        print(f"   {self.venv_path}")
        
        print("\n🔧 Development Commands:")
        print("   • Install new packages: pip install <package>")
        print("   • Update requirements: pip freeze > requirements.txt")
        print("   • Remove environment: rm -rf emoscan_env")
        
        print("\n" + "=" * 60)
    
    def setup(self):
        """Main setup process"""
        self.print_header()
        
        # Check prerequisites
        if not self.check_python_version():
            return False
        
        if not self.check_venv_module():
            return False
        
        # Create virtual environment
        if not self.create_virtual_environment():
            return False
        
        # Upgrade pip
        if not self.upgrade_pip():
            return False
        
        # Install requirements
        if not self.install_requirements():
            return False
        
        # Create helper scripts
        self.create_activation_scripts()
        self.create_run_scripts()
        
        # Print instructions
        self.print_usage_instructions()
        
        return True

def main():
    """Main function"""
    setup = VenvSetup()
    
    try:
        success = setup.setup()
        if success:
            print("✅ Setup completed successfully!")
            sys.exit(0)
        else:
            print("❌ Setup failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n❌ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
