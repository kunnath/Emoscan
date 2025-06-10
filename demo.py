#!/usr/bin/env python3
"""
Simple demo script to test camera connectivity and basic functionality
Run this before using the main Streamlit app to verify everything works
"""

import cv2
import sys
import time
from camera_handler import CameraHandler

def test_webcam():
    """Test local webcam connectivity"""
    print("🎥 Testing webcam connectivity...")
    
    camera_handler = CameraHandler()
    cap = camera_handler.initialize_camera("webcam")
    
    if cap is None:
        print("❌ Failed to connect to webcam")
        return False
    
    print("✅ Webcam connected successfully")
    
    # Test frame capture
    ret, frame = cap.read()
    if ret:
        print(f"✅ Frame captured: {frame.shape}")
        camera_info = camera_handler.get_camera_info(cap)
        print(f"📊 Camera info: {camera_info}")
    else:
        print("❌ Failed to capture frame")
        cap.release()
        return False
    
    cap.release()
    return True

def test_ip_camera(url):
    """Test IP camera connectivity"""
    print(f"🌐 Testing IP camera: {url}")
    
    camera_handler = CameraHandler()
    cap = camera_handler.initialize_camera(url)
    
    if cap is None:
        print("❌ Failed to connect to IP camera")
        return False
    
    print("✅ IP camera connected successfully")
    
    # Test frame capture
    ret, frame = cap.read()
    if ret:
        print(f"✅ Frame captured: {frame.shape}")
    else:
        print("❌ Failed to capture frame")
        cap.release()
        return False
    
    cap.release()
    return True

def test_emotion_detection():
    """Test emotion detection functionality"""
    print("😊 Testing emotion detection...")
    
    try:
        from emotion_detector import EmotionDetector
        detector = EmotionDetector()
        print("✅ Emotion detector initialized")
        return True
    except ImportError as e:
        print(f"❌ Failed to import emotion detector: {e}")
        return False
    except Exception as e:
        print(f"❌ Error initializing emotion detector: {e}")
        return False

def test_pose_detection():
    """Test pose detection functionality"""
    print("🤸 Testing pose detection...")
    
    try:
        from body_language_analyzer import BodyLanguageAnalyzer
        analyzer = BodyLanguageAnalyzer()
        print("✅ Pose analyzer initialized")
        return True
    except ImportError as e:
        print(f"❌ Failed to import pose analyzer: {e}")
        return False
    except Exception as e:
        print(f"❌ Error initializing pose analyzer: {e}")
        return False

def test_dependencies():
    """Test all required dependencies"""
    print("📦 Testing dependencies...")
    
    dependencies = [
        ('cv2', 'OpenCV'),
        ('streamlit', 'Streamlit'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
    ]
    
    optional_dependencies = [
        ('deepface', 'DeepFace'),
        ('mediapipe', 'MediaPipe'),
        ('plotly', 'Plotly'),
    ]
    
    all_good = True
    
    # Test required dependencies
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✅ {name} available")
        except ImportError:
            print(f"❌ {name} not available")
            all_good = False
    
    # Test optional dependencies
    for module, name in optional_dependencies:
        try:
            __import__(module)
            print(f"✅ {name} available")
        except ImportError:
            print(f"⚠️  {name} not available (optional)")
    
    return all_good

def interactive_demo():
    """Run interactive demo with live camera feed"""
    print("\n🎬 Starting interactive demo...")
    print("Press 'q' to quit, 's' to save frame, 'i' for info")
    
    camera_handler = CameraHandler()
    cap = camera_handler.initialize_camera("webcam")
    
    if cap is None:
        print("❌ Cannot start demo - no camera available")
        return
    
    try:
        from emotion_detector import EmotionDetector
        from body_language_analyzer import BodyLanguageAnalyzer
        
        emotion_detector = EmotionDetector()
        pose_analyzer = BodyLanguageAnalyzer()
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to capture frame")
                break
            
            frame_count += 1
            
            # Process every 10th frame for demo
            if frame_count % 10 == 0:
                # Analyze emotions
                emotion_result = emotion_detector.detect_emotions(frame)
                if emotion_result:
                    frame = emotion_detector.draw_annotations(frame, emotion_result)
                    print(f"😊 Emotion: {emotion_result['dominant_emotion']} "
                          f"({emotion_result['confidence']:.2f})")
                
                # Analyze posture
                posture_result = pose_analyzer.analyze_posture(frame)
                if posture_result:
                    frame = pose_analyzer.draw_annotations(frame, posture_result)
                    print(f"🤸 Posture: {posture_result['posture_type']} "
                          f"({posture_result['confidence']:.2f})")
            
            # Display frame
            cv2.imshow('EmoScan Demo - Press q to quit', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"emoscan_frame_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"💾 Frame saved as {filename}")
            elif key == ord('i'):
                info = camera_handler.get_camera_info(cap)
                print(f"📊 Camera info: {info}")
    
    except ImportError as e:
        print(f"⚠️  Running basic demo without AI analysis: {e}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            cv2.imshow('EmoScan Demo - Basic Camera Test', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("🎬 Demo ended")

def main():
    """Main demo function"""
    print("🎭 EmoScan Demo & Diagnostics")
    print("=" * 40)
    
    # Test dependencies
    if not test_dependencies():
        print("\n❌ Some required dependencies are missing.")
        print("Please run: pip install -r requirements.txt")
        return
    
    print("\n" + "=" * 40)
    
    # Test camera
    if not test_webcam():
        print("\n❌ Webcam test failed. Please check your camera.")
        
        # Ask for IP camera
        ip_url = input("\n🌐 Enter IP camera URL (or press Enter to skip): ").strip()
        if ip_url and not test_ip_camera(ip_url):
            print("❌ IP camera test also failed.")
            return
    
    print("\n" + "=" * 40)
    
    # Test AI components
    emotion_ok = test_emotion_detection()
    pose_ok = test_pose_detection()
    
    if not emotion_ok or not pose_ok:
        print("\n⚠️  Some AI components are not available.")
        print("The app will still work with basic camera functionality.")
    
    print("\n" + "=" * 40)
    print("✅ Basic tests completed!")
    
    # Ask for interactive demo
    demo = input("\n🎬 Run interactive demo? (y/N): ").strip().lower()
    if demo in ['y', 'yes']:
        interactive_demo()
    
    print("\n🚀 Ready to run the main app:")
    print("   streamlit run app.py")

if __name__ == "__main__":
    main()
