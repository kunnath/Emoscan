#!/usr/bin/env python3
"""
Test script to verify the Emoscan installation is working properly.
"""

import sys
import subprocess

def test_imports():
    """Test all required imports"""
    print("Testing package imports...")
    
    try:
        import streamlit
        print("✓ Streamlit")
    except ImportError as e:
        print(f"✗ Streamlit: {e}")
        return False
    
    try:
        import cv2
        print("✓ OpenCV")
    except ImportError as e:
        print(f"✗ OpenCV: {e}")
        return False
        
    try:
        import mediapipe
        print("✓ MediaPipe")
    except ImportError as e:
        print(f"✗ MediaPipe: {e}")
        return False
        
    try:
        from deepface import DeepFace
        print("✓ DeepFace")
    except ImportError as e:
        print(f"✗ DeepFace: {e}")
        return False
        
    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import plotly.express as px
        print("✓ Data science packages (NumPy, Pandas, Matplotlib, Seaborn, Plotly)")
    except ImportError as e:
        print(f"✗ Data science packages: {e}")
        return False
        
    return True

def test_versions():
    """Test package versions"""
    print("\nPackage versions:")
    
    import streamlit
    print(f"Streamlit: {streamlit.__version__}")
    
    import cv2
    print(f"OpenCV: {cv2.__version__}")
    
    import mediapipe
    print(f"MediaPipe: {mediapipe.__version__}")
    
    import tensorflow
    print(f"TensorFlow: {tensorflow.__version__}")
    
    import numpy
    print(f"NumPy: {numpy.__version__}")

def test_basic_functionality():
    """Test basic functionality"""
    print("\nTesting basic functionality...")
    
    try:
        import cv2
        import mediapipe as mp
        
        # Test MediaPipe Face Detection
        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection()
        print("✓ MediaPipe face detection initialized")
        
        # Test OpenCV
        # Create a simple test image
        import numpy as np
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        print("✓ OpenCV color conversion working")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 50)
    print("EMOSCAN INSTALLATION TEST")
    print("=" * 50)
    
    # Test Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("⚠️  Warning: Python 3.8+ recommended")
    else:
        print("✓ Python version is compatible")
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed!")
        return False
    
    # Test versions
    test_versions()
    
    # Test basic functionality
    if not test_basic_functionality():
        print("\n❌ Functionality tests failed!")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED!")
    print("Your Emoscan installation is ready to use.")
    print("\nTo start the application, run:")
    print("  streamlit run app.py")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
