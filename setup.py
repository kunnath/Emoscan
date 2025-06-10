#!/usr/bin/env python3
"""
Setup script for EmoScan application
Handles installation, configuration, and environment setup
"""

import os
import sys
import subprocess
import platform
import pkg_resources
from pathlib import Path

def check_python_version():
    """Check if Python version meets requirements"""
    min_version = (3, 8)
    current_version = sys.version_info[:2]
    
    print(f"🐍 Python version: {sys.version}")
    
    if current_version < min_version:
        print(f"❌ Python {min_version[0]}.{min_version[1]}+ required")
        return False
    
    print("✅ Python version OK")
    return True

def check_system_resources():
    """Check system resources"""
    import psutil
    
    # Check RAM
    total_ram = psutil.virtual_memory().total / (1024**3)  # GB
    print(f"💾 System RAM: {total_ram:.1f} GB")
    
    if total_ram < 4:
        print("⚠️  Low RAM detected. 4GB+ recommended for optimal performance")
    else:
        print("✅ RAM OK")
    
    # Check disk space
    disk_usage = psutil.disk_usage('.')
    free_space = disk_usage.free / (1024**3)  # GB
    print(f"💿 Free disk space: {free_space:.1f} GB")
    
    if free_space < 2:
        print("⚠️  Low disk space. 2GB+ required for models and data")
    else:
        print("✅ Disk space OK")

def install_package(package_name, pip_name=None):
    """Install a Python package using pip"""
    if pip_name is None:
        pip_name = package_name
    
    try:
        pkg_resources.get_distribution(package_name)
        print(f"✅ {package_name} already installed")
        return True
    except pkg_resources.DistributionNotFound:
        print(f"📦 Installing {package_name}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
            print(f"✅ {package_name} installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package_name}: {e}")
            return False

def install_requirements():
    """Install all required packages"""
    print("\n📦 Installing required packages...")
    
    requirements_file = Path("requirements.txt")
    
    if requirements_file.exists():
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ])
            print("✅ All requirements installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install requirements: {e}")
            return False
    else:
        print("⚠️  requirements.txt not found. Installing individual packages...")
        
        # Core packages
        packages = [
            ("streamlit", "streamlit>=1.28.1"),
            ("cv2", "opencv-python>=4.8.1.78"),
            ("numpy", "numpy>=1.24.3"),
            ("pandas", "pandas>=2.0.3"),
            ("PIL", "pillow>=10.0.1")
        ]
        
        success = True
        for package_name, pip_name in packages:
            if not install_package(package_name, pip_name):
                success = False
        
        return success

def install_optional_packages():
    """Install optional AI packages"""
    print("\n🧠 Installing AI/ML packages...")
    
    # Check for Apple Silicon Mac
    is_apple_silicon = platform.machine() == 'arm64' and platform.system() == 'Darwin'
    
    if is_apple_silicon:
        print("🍎 Detected Apple Silicon Mac - using optimized packages")
        ai_packages = [
            ("tensorflow", "tensorflow-macos"),
            ("deepface", "deepface"),
            ("mediapipe", "mediapipe"),
            ("plotly", "plotly")
        ]
    else:
        ai_packages = [
            ("tensorflow", "tensorflow"),
            ("deepface", "deepface"),
            ("mediapipe", "mediapipe"),
            ("plotly", "plotly")
        ]
    
    for package_name, pip_name in ai_packages:
        install_package(package_name, pip_name)

def setup_directories():
    """Create necessary directories"""
    print("\n📁 Setting up directories...")
    
    directories = ['data', 'models', 'exports', 'logs']
    
    for dir_name in directories:
        dir_path = Path(dir_name)
        dir_path.mkdir(exist_ok=True)
        print(f"✅ Created {dir_name}/ directory")

def test_camera_access():
    """Test camera access"""
    print("\n📷 Testing camera access...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print("✅ Camera access OK")
                print(f"📊 Camera resolution: {frame.shape[1]}x{frame.shape[0]}")
            else:
                print("⚠️  Camera detected but unable to capture frames")
            cap.release()
        else:
            print("⚠️  No camera detected or camera access denied")
            print("💡 Make sure:")
            print("   - Camera is connected")
            print("   - Camera permissions are granted")
            print("   - No other app is using the camera")
    
    except ImportError:
        print("⚠️  OpenCV not available - skipping camera test")

def test_ai_models():
    """Test AI model loading"""
    print("\n🧠 Testing AI models...")
    
    # Test DeepFace
    try:
        from deepface import DeepFace
        print("✅ DeepFace available")
        
        # Try to analyze a small test image
        import numpy as np
        test_image = np.zeros((48, 48, 3), dtype=np.uint8)
        try:
            DeepFace.analyze(test_image, actions=['emotion'], enforce_detection=False, silent=True)
            print("✅ DeepFace emotion model loaded")
        except Exception as e:
            print(f"⚠️  DeepFace model loading issue: {e}")
    
    except ImportError:
        print("⚠️  DeepFace not available")
    
    # Test MediaPipe
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose()
        print("✅ MediaPipe pose model loaded")
        pose.close()
    except ImportError:
        print("⚠️  MediaPipe not available")
    except Exception as e:
        print(f"⚠️  MediaPipe model loading issue: {e}")

def create_demo_config():
    """Create demo configuration file"""
    print("\n⚙️  Creating demo configuration...")
    
    config_content = """
# EmoScan Demo Configuration
# Copy this to config_local.py and modify as needed

DEMO_CONFIG = {
    'camera_source': 'webcam',  # 'webcam', 'ip_camera', or URL
    'ip_camera_url': 'http://192.168.1.100:8080/video',
    'enable_emotion_analysis': True,
    'enable_posture_analysis': True,
    'save_analysis_data': False,
    'auto_start': False
}

# Phone camera URLs for quick reference
PHONE_CAMERA_EXAMPLES = {
    'android_ip_webcam': 'http://192.168.1.XXX:8080/video',
    'android_droidcam': 'http://192.168.1.XXX:4747/mjpegfeed',
    'ios_epoccam': 'http://192.168.1.XXX:80/live'
}
"""
    
    with open('config_demo.py', 'w') as f:
        f.write(config_content)
    
    print("✅ Demo configuration created (config_demo.py)")

def main():
    """Main setup function"""
    print("🎭 EmoScan Setup Script")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        print("\n❌ Setup failed: Python version too old")
        sys.exit(1)
    
    # Check system resources
    try:
        check_system_resources()
    except ImportError:
        print("⚠️  psutil not available - skipping resource check")
        print("💡 Install psutil for system monitoring: pip install psutil")
    
    # Install packages
    print("\n" + "=" * 50)
    if not install_requirements():
        print("\n❌ Setup failed: Could not install required packages")
        sys.exit(1)
    
    # Install optional packages
    install_optional_packages()
    
    # Setup directories
    setup_directories()
    
    # Test functionality
    print("\n" + "=" * 50)
    print("🧪 Testing functionality...")
    
    test_camera_access()
    test_ai_models()
    
    # Create demo config
    create_demo_config()
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed!")
    print("\n🚀 Next steps:")
    print("1. Run the demo: python demo.py")
    print("2. Start the app: streamlit run app.py")
    print("3. Configure camera in the sidebar")
    print("4. Click 'Start Analysis' to begin")
    
    print("\n💡 Tips:")
    print("- For phone cameras, install IP Webcam (Android) or EpocCam (iOS)")
    print("- Make sure phone and computer are on the same WiFi network")
    print("- Grant camera permissions if prompted")
    print("- Check README.md for detailed instructions")

if __name__ == "__main__":
    main()
