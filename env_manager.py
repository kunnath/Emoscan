#!/usr/bin/env python3
"""
EmoScan Environment Manager
Provides utilities for managing the virtual environment
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path

class EnvManager:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.venv_name = "emoscan_env"
        self.venv_path = self.project_root / self.venv_name
        self.requirements_file = self.project_root / "requirements.txt"
    
    def status(self):
        """Show environment status"""
        print("🎭 EmoScan Environment Status")
        print("=" * 50)
        
        # Check if venv exists
        if self.venv_path.exists():
            print(f"✅ Virtual environment: {self.venv_path}")
            
            # Check if activated
            if os.environ.get('VIRTUAL_ENV') == str(self.venv_path):
                print("✅ Environment is currently activated")
            else:
                print("⚠️  Environment is not activated")
            
            # Check installed packages
            try:
                pip_exe = self.get_pip_executable()
                result = subprocess.run([
                    str(pip_exe), "list", "--format=freeze"
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    packages = result.stdout.strip().split('\n')
                    print(f"📦 Installed packages: {len(packages)}")
                    
                    # Check key packages
                    key_packages = ['streamlit', 'opencv-python', 'mediapipe', 'deepface']
                    for pkg in key_packages:
                        found = any(line.lower().startswith(pkg.lower()) for line in packages)
                        status = "✅" if found else "❌"
                        print(f"   {status} {pkg}")
                
            except Exception as e:
                print(f"⚠️  Could not check packages: {e}")
        else:
            print("❌ Virtual environment not found")
            print(f"   Expected location: {self.venv_path}")
        
        print("\n🐍 Python Information:")
        print(f"   Version: {sys.version}")
        print(f"   Executable: {sys.executable}")
        print(f"   Platform: {platform.platform()}")
    
    def get_pip_executable(self):
        """Get pip executable path"""
        if platform.system() == "Windows":
            return self.venv_path / "Scripts" / "pip.exe"
        else:
            return self.venv_path / "bin" / "pip"
    
    def get_python_executable(self):
        """Get Python executable path"""
        if platform.system() == "Windows":
            return self.venv_path / "Scripts" / "python.exe"
        else:
            return self.venv_path / "bin" / "python"
    
    def clean(self):
        """Clean the virtual environment"""
        if self.venv_path.exists():
            print(f"🗑️  Removing virtual environment: {self.venv_path}")
            shutil.rmtree(self.venv_path)
            print("✅ Environment removed")
        else:
            print("ℹ️  No virtual environment to clean")
    
    def reinstall(self):
        """Reinstall packages"""
        if not self.venv_path.exists():
            print("❌ Virtual environment not found. Run setup first.")
            return False
        
        print("🔄 Reinstalling packages...")
        try:
            subprocess.run([
                str(self.get_pip_executable()), "install", "--force-reinstall", 
                "-r", str(self.requirements_file)
            ], check=True)
            print("✅ Packages reinstalled successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to reinstall packages: {e}")
            return False
    
    def update(self):
        """Update all packages to latest versions"""
        if not self.venv_path.exists():
            print("❌ Virtual environment not found. Run setup first.")
            return False
        
        print("📦 Updating packages...")
        try:
            # Get installed packages
            result = subprocess.run([
                str(self.get_pip_executable()), "list", "--outdated", "--format=json"
            ], capture_output=True, text=True, check=True)
            
            import json
            outdated_packages = json.loads(result.stdout)
            
            if not outdated_packages:
                print("✅ All packages are up to date")
                return True
            
            print(f"📦 Found {len(outdated_packages)} outdated packages")
            
            for package in outdated_packages:
                print(f"   Updating {package['name']} {package['version']} -> {package['latest_version']}")
                subprocess.run([
                    str(self.get_pip_executable()), "install", "--upgrade", package['name']
                ], check=True)
            
            print("✅ All packages updated successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to update packages: {e}")
            return False
        except json.JSONDecodeError:
            print("⚠️  Could not parse package list")
            return False
    
    def export_requirements(self):
        """Export current requirements to file"""
        if not self.venv_path.exists():
            print("❌ Virtual environment not found.")
            return False
        
        try:
            result = subprocess.run([
                str(self.get_pip_executable()), "freeze"
            ], capture_output=True, text=True, check=True)
            
            with open(self.requirements_file, 'w') as f:
                f.write(result.stdout)
            
            print(f"✅ Requirements exported to {self.requirements_file}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to export requirements: {e}")
            return False
    
    def shell(self):
        """Open a shell with the environment activated"""
        if not self.venv_path.exists():
            print("❌ Virtual environment not found. Run setup first.")
            return False
        
        print("🐚 Opening shell with activated environment...")
        print("   Type 'exit' to return to normal shell")
        
        # Set environment variables
        env = os.environ.copy()
        env['VIRTUAL_ENV'] = str(self.venv_path)
        
        if platform.system() == "Windows":
            env['PATH'] = str(self.venv_path / "Scripts") + os.pathsep + env.get('PATH', '')
            subprocess.run(['cmd'], env=env)
        else:
            env['PATH'] = str(self.venv_path / "bin") + os.pathsep + env.get('PATH', '')
            subprocess.run([os.environ.get('SHELL', '/bin/bash')], env=env)
        
        return True

def main():
    """Main function"""
    manager = EnvManager()
    
    if len(sys.argv) < 2:
        print("🎭 EmoScan Environment Manager")
        print("\nUsage: python env_manager.py <command>")
        print("\nCommands:")
        print("  status     - Show environment status")
        print("  clean      - Remove virtual environment")
        print("  reinstall  - Reinstall all packages")
        print("  update     - Update all packages")
        print("  export     - Export requirements.txt")
        print("  shell      - Open shell with environment activated")
        return
    
    command = sys.argv[1].lower()
    
    if command == "status":
        manager.status()
    elif command == "clean":
        manager.clean()
    elif command == "reinstall":
        manager.reinstall()
    elif command == "update":
        manager.update()
    elif command == "export":
        manager.export_requirements()
    elif command == "shell":
        manager.shell()
    else:
        print(f"❌ Unknown command: {command}")
        print("   Run without arguments to see available commands")

if __name__ == "__main__":
    main()
