# EmoScan - Emotion & Body Language Analyzer 🎭

A real-time Streamlit application that analyzes facial emotions and body language patterns using computer vision and machine learning. Supports webcam, WiFi cameras, and phone cameras.

![EmoScan Application](emo.png)

![EmoScan Demo](https://img.shields.io/badge/Status-Ready-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red)
![Camera](https://img.shields.io/badge/Camera-Enhanced-green)

## 🆕 Latest Updates (June 2025)

### Camera Robustness Enhancement
We've significantly improved camera reliability and user experience:

- **🔄 Automatic Reconnection**: Camera automatically reconnects on failures
- **📊 Real-time Health Monitoring**: Live connection status indicators
- **🛡️ Persistent Recording**: Camera only stops when you command it to stop
- **🔧 Enhanced Error Recovery**: Smart retry mechanisms with progressive backoff
- **📈 Performance Boost**: Improved frame processing for better responsiveness
- **🎛️ Better Controls**: New restart and test connection features
- **💡 Smart Diagnostics**: Detailed error messages with recovery suggestions

## 🚀 Features

- **Real-time Emotion Detection**: Analyze facial expressions using DeepFace
- **Body Language Analysis**: Track posture and gestures using MediaPipe
- **Multi-Camera Support**: Webcam, WiFi/IP cameras, and smartphone cameras
- **Enhanced Camera Robustness**: Automatic reconnection and error recovery
- **Live Visualizations**: Real-time charts and statistics
- **Data Export**: Save analysis results to CSV
- **Comprehensive Dashboard**: Combined emotion and posture insights
- **Camera Health Monitoring**: Real-time connection status and diagnostics
- **Persistent Recording**: Camera continues until explicitly stopped by user

## 📋 Requirements

### System Requirements
- Python 3.8 or higher
- macOS, Windows, or Linux
- Camera (built-in, USB, or network camera)
- Minimum 4GB RAM recommended

### Hardware Compatibility
- **Webcam**: Any USB or built-in camera
- **WiFi Cameras**: IP cameras with MJPEG or RTSP streams
- **Phone Cameras**: Using apps like IP Webcam (Android) or EpocCam (iOS)

## 🔧 Installation

### Option 1: Automated Setup (Recommended)
The easiest way to get started is using our automated setup script:

```bash
# Clone the repository
git clone <your-repo-url>
cd Emoscan

# Run the automated virtual environment setup
python3 setup_venv.py
```

This will:
- ✅ Create a virtual environment
- ✅ Install all dependencies
- ✅ Create activation and run scripts
- ✅ Verify system compatibility

### Option 2: Manual Setup

#### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd Emoscan
```

#### 2. Create Virtual Environment
```bash
python3 -m venv emoscan_env
```

#### 3. Activate Virtual Environment
**macOS/Linux:**
```bash
source emoscan_env/bin/activate
```

**Windows:**
```bash
emoscan_env\Scripts\activate
```

#### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 🚀 Quick Start

After installation, you can start the application in several ways:

### Easy Way (Automated Setup)
If you used the automated setup, simply run:
```bash
# Activate environment and run app
./run_app.sh        # macOS/Linux
# or
run_app.bat         # Windows
```

### Manual Way
```bash
# 1. Activate virtual environment
source emoscan_env/bin/activate  # macOS/Linux
# emoscan_env\Scripts\activate   # Windows

# 2. Start the application
streamlit run app.py
```

The application will open in your web browser at `http://localhost:8501`

### 4. Download Required Models
The application will automatically download required models on first run:
- DeepFace emotion recognition models
- MediaPipe pose estimation models

## 🎮 Usage

### Starting the Application
```bash
streamlit run app.py
```

The application will open in your web browser at `http://localhost:8501`

### Camera Setup

#### Webcam
1. Select "Webcam" in the sidebar
2. Click "Start Analysis"
3. **New**: Camera will continue running until you click "Stop Analysis"

#### WiFi/IP Camera
1. Select "WiFi Camera (IP)" in the sidebar
2. Enter your camera's IP address and port
3. Common formats:
   - `http://192.168.1.100:8080/video`
   - `rtsp://192.168.1.100:554/stream`
4. **New**: Automatic reconnection on connection failures

#### Phone Camera
1. Install IP Webcam app on your phone:
   - **Android**: [IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam)
   - **iOS**: [EpocCam](https://apps.apple.com/app/epoccam-webcam-for-computer/id435355256)

2. Connect phone to same WiFi network
3. Start the camera server in the app
4. Select "Phone Camera" and enter the displayed URL

### Enhanced Camera Controls (New!)

#### Camera Status Monitoring
- **Real-time Status**: Live connection health indicator (Good/Fair/Poor)
- **Frame Counter**: Track processed frames in real-time
- **Connection Test**: Quick camera connection verification
- **Auto-Recovery**: Automatic reconnection on failures

#### Control Buttons
- **Start Analysis**: Begin camera recording and analysis
- **Stop Analysis**: Manually stop camera (disabled during recording for safety)
- **Restart Camera**: Quick camera restart without losing settings
- **Test Connection**: Verify camera connectivity before starting

#### Robust Error Handling
- **Automatic Reconnection**: Up to 5 reconnection attempts with progressive delays
- **Persistent Recording**: Camera won't stop unexpectedly - only when user commands
- **Error Recovery**: Detailed error messages with recovery suggestions
- **Connection Health**: Real-time monitoring of camera connection quality

### Configuration Options

#### Analysis Settings
- **Emotion Sensitivity**: Adjust emotion detection threshold (0.1-1.0)
- **Posture Sensitivity**: Adjust pose detection threshold (0.1-1.0)
- **Analysis Options**: Enable/disable emotion or posture analysis
- **Data Saving**: Option to save analysis data for export

#### Camera Optimization
The app automatically optimizes camera settings for:
- Resolution (640x480 for performance)
- Frame rate (30 FPS for local, 15 FPS for network cameras)
- Image enhancement for better detection
- **New**: Multiple backend support for better compatibility
- **New**: Advanced buffer management for network cameras
- **New**: Persistent connection monitoring and auto-recovery

## 📊 Features Overview

### Emotion Analysis
- **Supported Emotions**: Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral
- **Real-time Detection**: Live emotion recognition with confidence scores
- **Emotion History**: Track emotion changes over time
- **Statistics**: Emotion distribution, confidence levels, and trends

### Body Language Analysis
- **Posture Types**:
  - Good Posture (Upright)
  - Slouched
  - Leaning Left/Right
  - Head Down
  - Arms Crossed
  - Hands on Hips
  - Relaxed

- **Posture Metrics**:
  - Posture score (0-100)
  - Shoulder alignment
  - Head position
  - Body symmetry

### Visualizations
- Real-time emotion timeline
- Posture quality gauge
- Emotion distribution charts
- Correlation analysis
- Statistical summaries

## 🛠️ Technical Details

### Core Technologies
- **Streamlit**: Web interface and real-time updates
- **OpenCV**: Video capture and image processing
- **DeepFace**: Facial emotion recognition
- **MediaPipe**: Pose estimation and body tracking
- **Plotly**: Interactive visualizations
- **Pandas**: Data analysis and export

### Architecture
```
app.py                 # Main Streamlit application
├── emotion_detector.py    # Emotion analysis using DeepFace
├── body_language_analyzer.py  # Posture analysis using MediaPipe
├── camera_handler.py      # Camera management and optimization
└── data_visualizer.py     # Charts and statistical analysis
```

### Performance Optimization
- **Enhanced**: Frame processing every 2nd frame for improved real-time performance
- Automatic resolution adjustment for network cameras
- **New**: Advanced buffering and retry mechanisms for unstable connections
- **New**: Multiple reconnection attempts with smart backoff
- Efficient data structures for historical analysis
- **New**: Camera health monitoring and automatic recovery

## 🔧 Troubleshooting

### Common Issues

#### Camera Connection Problems
```bash
# Check available cameras
python -c "import cv2; print([i for i in range(10) if cv2.VideoCapture(i).read()[0]])"
```

**New Camera Robustness Features:**
- **Automatic Recovery**: Camera will attempt to reconnect automatically
- **Health Monitoring**: Check camera status in sidebar
- **Test Connection**: Use "Test Camera Connection" button before starting
- **Restart Camera**: Use "Restart Camera" button if issues persist
- **Multiple Backends**: App tries different camera backends automatically

#### Camera Stops Unexpectedly (Fixed!)
**Previous Issue**: Camera would stop due to connection failures or errors
**Solution Implemented**:
- Enhanced error handling with automatic reconnection
- Persistent recording state that won't stop unexpectedly
- Multiple retry mechanisms for failed reads
- Progressive backoff for reconnection attempts
- Real-time connection health monitoring

**If camera still stops**:
1. Check the camera health indicator in sidebar
2. Use "Test Camera Connection" to verify setup
3. Click "Restart Camera" to force reconnection
4. Check error messages for specific recovery suggestions

#### Permission Issues (macOS)
- Go to System Preferences → Security & Privacy → Camera
- Grant permission to Terminal or your Python environment

#### Network Camera Issues
- Test camera URL in web browser first
- Check firewall settings
- Ensure camera and computer are on same network
- Try different stream formats (MJPEG vs RTSP)

#### Performance Issues
- Lower camera resolution in camera_handler.py
- Reduce analysis frequency by modifying frame_count conditions
- Close other applications using camera
- Ensure adequate RAM (4GB+ recommended)

### Installation Issues

#### TensorFlow/DeepFace Installation
```bash
# For Apple Silicon Macs
pip install tensorflow-macos tensorflow-metal

# For CUDA-enabled GPUs
pip install tensorflow-gpu
```

#### MediaPipe Installation Issues
```bash
# Alternative installation
pip install mediapipe-silicon  # For Apple Silicon
pip install mediapipe  # For other systems
```

## 📈 Advanced Configuration

### Custom Emotion Models
Modify `emotion_detector.py` to use custom models:
```python
# Replace DeepFace with custom model
result = your_custom_model.predict(face_roi)
```

### Custom Posture Classifications
Extend posture types in `body_language_analyzer.py`:
```python
self.posture_types = {
    'custom_posture': 'Custom Description',
    # Add your posture types
}
```

### Export Settings
Configure data export in `data_visualizer.py`:
```python
# Customize export fields
export_fields = ['timestamp', 'emotion', 'posture_score']
```

## 📊 Data Analysis

### Export Options
- **CSV Export**: Complete analysis data with timestamps
- **Statistical Reports**: Emotion and posture summaries
- **Visualization Export**: Save charts as images

### Data Structure
```json
{
  "timestamp": "2024-01-01T10:00:00",
  "emotions": {
    "dominant_emotion": "happy",
    "confidence": 0.85,
    "all_emotions": {"happy": 0.85, "neutral": 0.15}
  },
  "posture": {
    "posture_type": "upright",
    "confidence": 0.92,
    "posture_score": 85
  }
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [DeepFace](https://github.com/serengil/deepface) for emotion recognition
- [MediaPipe](https://mediapipe.dev/) for pose estimation
- [Streamlit](https://streamlit.io/) for the web interface
- [OpenCV](https://opencv.org/) for computer vision

## 📞 Support

For issues and questions:
1. Check the troubleshooting section above
2. Search existing [GitHub Issues](issues)
3. Create a new issue with detailed description

## 📝 Changelog

### Version 2.1.0 (June 10, 2025) - Camera Robustness Update
#### 🚀 New Features
- **Enhanced Camera Reliability**: Implemented robust camera connection management
- **Automatic Reconnection**: Camera automatically reconnects on connection failures (up to 5 attempts)
- **Real-time Health Monitoring**: Live camera connection status indicators (Good/Fair/Poor)
- **Persistent Recording**: Camera continues recording until explicitly stopped by user
- **Smart Error Recovery**: Progressive backoff and intelligent retry mechanisms

#### 🔧 Improvements
- **Performance Boost**: Reduced frame processing interval for better responsiveness
- **Better UI Controls**: Added "Restart Camera" and "Test Connection" buttons
- **Enhanced Error Messages**: Detailed error descriptions with recovery suggestions
- **Session State Management**: Improved handling of Streamlit session state
- **Multiple Backend Support**: Automatic fallback to different camera backends

#### 🛠️ Technical Enhancements
- Increased maximum consecutive failures threshold (10 → 15)
- Added camera health tracking and metrics
- Implemented multiple reconnection attempts with smart delays
- Enhanced IP camera initialization with multiple successful read requirements
- Added buffer management for network cameras
- Improved webcam initialization with backend fallback

#### 🐛 Bug Fixes
- Fixed camera stopping unexpectedly without user command
- Resolved session state corruption issues
- Fixed IP camera connection timeout problems
- Improved handling of frame processing errors

### Version 2.0.0 (Previous Release)
- Initial release with emotion and body language analysis
- Multi-camera support (webcam, IP cameras, phone cameras)
- Real-time visualizations and data export
- Comprehensive analysis dashboard

## 🔮 Roadmap

### ✅ Recently Completed
- [x] Enhanced camera robustness and automatic reconnection
- [x] Real-time camera health monitoring
- [x] Persistent recording state management
- [x] Advanced error recovery mechanisms
- [x] Improved user interface controls

### 🚧 In Progress
- [ ] Multiple person detection
- [ ] Voice emotion analysis

### 📋 Planned Features
- [ ] Cloud deployment options
- [ ] Mobile app version
- [ ] Integration with fitness trackers
- [ ] Advanced gesture recognition
- [ ] Real-time alerts and notifications
- [ ] Weapon detection and security features
- [ ] Enhanced data visualization dashboards

---

**Made with ❤️ for emotion and behavior analysis**
# Emoscan
