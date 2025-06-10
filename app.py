import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from collections import deque
import threading
import queue

# Import custom modules
from emotion_detector import EmotionDetector
from body_language_analyzer import BodyLanguageAnalyzer
from camera_handler import CameraHandler
from data_visualizer import DataVisualizer

# Page configuration
st.set_page_config(
    page_title="EmoScan - Emotion & Body Language Analyzer",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .emotion-display {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    .sidebar-section {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class EmoscanApp:
    def __init__(self):
        self.emotion_detector = EmotionDetector()
        self.body_analyzer = BodyLanguageAnalyzer()
        self.camera_handler = CameraHandler()
        self.visualizer = DataVisualizer()
        
        # Initialize session state
        if 'analysis_data' not in st.session_state:
            st.session_state.analysis_data = []
        if 'is_recording' not in st.session_state:
            st.session_state.is_recording = False
        if 'camera_source' not in st.session_state:
            st.session_state.camera_source = "webcam"
        if 'emotion_history' not in st.session_state:
            st.session_state.emotion_history = deque(maxlen=100)
        if 'posture_history' not in st.session_state:
            st.session_state.posture_history = deque(maxlen=100)

    def render_sidebar(self):
        with st.sidebar:
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.header("🎯 Configuration")
            
            # Camera source selection
            camera_type = st.selectbox(
                "📷 Camera Source",
                ["Webcam", "WiFi Camera (IP)", "Phone Camera"],
                index=0
            )
            
            if camera_type == "WiFi Camera (IP)":
                ip_address = st.text_input("IP Address", "192.168.1.100")
                port = st.text_input("Port", "8080")
                stream_path = st.text_input("Stream Path", "/video")
                st.session_state.camera_source = f"http://{ip_address}:{port}{stream_path}"
            elif camera_type == "Phone Camera":
                st.info("📱 Install 'IP Webcam' app on your phone and enter the URL below")
                phone_url = st.text_input("Phone Camera URL", "http://192.168.1.101:8080/video")
                st.session_state.camera_source = phone_url
            else:
                st.session_state.camera_source = "webcam"
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Analysis settings
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.header("⚙️ Analysis Settings")
            
            emotion_sensitivity = st.slider("Emotion Sensitivity", 0.1, 1.0, 0.7)
            posture_sensitivity = st.slider("Posture Sensitivity", 0.1, 1.0, 0.6)
            
            analyze_emotions = st.checkbox("Analyze Emotions", True)
            analyze_posture = st.checkbox("Analyze Body Language", True)
            save_data = st.checkbox("Save Analysis Data", False)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Real-time stats
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.header("📊 Live Stats")
            
            if st.session_state.emotion_history:
                recent_emotions = list(st.session_state.emotion_history)[-10:]
                if recent_emotions:
                    dominant_emotion = max(set(recent_emotions), key=recent_emotions.count)
                    st.metric("Dominant Emotion", dominant_emotion)
            
            if st.session_state.posture_history:
                recent_postures = list(st.session_state.posture_history)[-10:]
                if recent_postures:
                    avg_confidence = np.mean([p['confidence'] for p in recent_postures if 'confidence' in p])
                    st.metric("Posture Confidence", f"{avg_confidence:.2f}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            return {
                'emotion_sensitivity': emotion_sensitivity,
                'posture_sensitivity': posture_sensitivity,
                'analyze_emotions': analyze_emotions,
                'analyze_posture': analyze_posture,
                'save_data': save_data
            }

    def render_main_interface(self, settings):
        st.markdown('<h1 class="main-header">🎭 EmoScan - Emotion & Body Language Analyzer</h1>', 
                   unsafe_allow_html=True)
        
        # Control buttons
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("▶️ Start Analysis", type="primary"):
                st.session_state.is_recording = True
                
        with col2:
            if st.button("⏹️ Stop Analysis"):
                st.session_state.is_recording = False
                
        with col3:
            if st.button("📊 Show Statistics"):
                self.show_statistics()
                
        with col4:
            if st.button("🗑️ Clear Data"):
                st.session_state.analysis_data = []
                st.session_state.emotion_history.clear()
                st.session_state.posture_history.clear()
                st.rerun()

        # Main content area
        if st.session_state.is_recording:
            self.run_live_analysis(settings)
        else:
            self.show_welcome_screen()

    def show_welcome_screen(self):
        st.markdown("""
        ## 🚀 Welcome to EmoScan!
        
        This application analyzes facial emotions and body language in real-time using advanced AI models.
        
        ### 📋 Features:
        - **Real-time emotion detection** using facial expression analysis
        - **Body language assessment** through pose estimation
        - **WiFi camera support** for remote monitoring
        - **Live data visualization** and statistics
        - **Historical analysis** and trends
        
        ### 🎯 How to use:
        1. Configure your camera source in the sidebar
        2. Adjust analysis settings
        3. Click "Start Analysis" to begin
        4. View real-time results and statistics
        
        **Ready to start? Click the "Start Analysis" button above!**
        """)

    def run_live_analysis(self, settings):
        # Create placeholder containers
        video_col, stats_col = st.columns([2, 1])
        
        with video_col:
            st.subheader("📹 Live Feed")
            video_placeholder = st.empty()
            
        with stats_col:
            st.subheader("📊 Real-time Analysis")
            emotion_placeholder = st.empty()
            posture_placeholder = st.empty()
            
        # Charts section
        st.subheader("📈 Live Charts")
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            emotion_chart_placeholder = st.empty()
            
        with chart_col2:
            posture_chart_placeholder = st.empty()

        # Initialize camera
        cap = self.camera_handler.initialize_camera(st.session_state.camera_source)
        
        if cap is None:
            st.error("❌ Failed to connect to camera. Please check your camera source.")
            st.session_state.is_recording = False
            return

        frame_count = 0
        
        try:
            while st.session_state.is_recording:
                ret, frame = cap.read()
                if not ret:
                    st.error("❌ Failed to read from camera")
                    break
                
                frame_count += 1
                
                # Process frame every few frames to improve performance
                if frame_count % 3 == 0:
                    analysis_result = self.analyze_frame(frame, settings)
                    
                    # Update displays
                    self.update_video_display(video_placeholder, frame, analysis_result)
                    self.update_stats_display(emotion_placeholder, posture_placeholder, analysis_result)
                    
                    # Update charts
                    if frame_count % 15 == 0:  # Update charts less frequently
                        self.update_charts(emotion_chart_placeholder, posture_chart_placeholder)
                
                # Add small delay to prevent overwhelming the interface
                time.sleep(0.1)
                
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
        finally:
            cap.release()

    def analyze_frame(self, frame, settings):
        results = {}
        timestamp = datetime.now()
        
        try:
            if settings['analyze_emotions']:
                emotion_result = self.emotion_detector.detect_emotions(
                    frame, 
                    sensitivity=settings['emotion_sensitivity']
                )
                results['emotions'] = emotion_result
                
                # Store emotion history
                if emotion_result and 'dominant_emotion' in emotion_result:
                    st.session_state.emotion_history.append(emotion_result['dominant_emotion'])
            
            if settings['analyze_posture']:
                posture_result = self.body_analyzer.analyze_posture(
                    frame,
                    sensitivity=settings['posture_sensitivity']
                )
                results['posture'] = posture_result
                
                # Store posture history
                if posture_result:
                    st.session_state.posture_history.append(posture_result)
            
            # Save data if enabled
            if settings['save_data']:
                analysis_data = {
                    'timestamp': timestamp,
                    'emotions': results.get('emotions', {}),
                    'posture': results.get('posture', {})
                }
                st.session_state.analysis_data.append(analysis_data)
            
        except Exception as e:
            st.error(f"Analysis error: {str(e)}")
            
        return results

    def update_video_display(self, placeholder, frame, analysis_result):
        # Draw analysis overlays on frame
        annotated_frame = frame.copy()
        
        # Draw emotion annotations
        if 'emotions' in analysis_result:
            annotated_frame = self.emotion_detector.draw_annotations(
                annotated_frame, 
                analysis_result['emotions']
            )
        
        # Draw posture annotations
        if 'posture' in analysis_result:
            annotated_frame = self.body_analyzer.draw_annotations(
                annotated_frame, 
                analysis_result['posture']
            )
        
        # Convert to RGB for Streamlit
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        placeholder.image(annotated_frame_rgb, channels="RGB", use_column_width=True)

    def update_stats_display(self, emotion_placeholder, posture_placeholder, analysis_result):
        # Update emotion display
        with emotion_placeholder:
            if 'emotions' in analysis_result and analysis_result['emotions']:
                emotions = analysis_result['emotions']
                if 'dominant_emotion' in emotions:
                    emotion_color = self.get_emotion_color(emotions['dominant_emotion'])
                    confidence = emotions.get('confidence', 0.0)
                    
                    st.markdown(f"""
                    <div class="emotion-display" style="background-color: {emotion_color};">
                        😊 {emotions['dominant_emotion'].title()}<br>
                        <small>Confidence: {confidence:.2f}</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show emotion breakdown
                    if 'all_emotions' in emotions:
                        for emotion, score in emotions['all_emotions'].items():
                            st.metric(emotion.title(), f"{score:.3f}")
            else:
                st.info("No emotions detected")
        
        # Update posture display
        with posture_placeholder:
            if 'posture' in analysis_result and analysis_result['posture']:
                posture = analysis_result['posture']
                
                st.markdown("### 🤸 Body Language")
                
                if 'posture_type' in posture:
                    st.metric("Posture", posture['posture_type'])
                
                if 'confidence' in posture:
                    st.metric("Confidence", f"{posture['confidence']:.2f}")
                
                if 'keypoints_detected' in posture:
                    st.metric("Keypoints", posture['keypoints_detected'])
                    
                if 'pose_analysis' in posture:
                    analysis = posture['pose_analysis']
                    for key, value in analysis.items():
                        st.metric(key.replace('_', ' ').title(), f"{value:.2f}")
            else:
                st.info("No pose detected")

    def update_charts(self, emotion_chart_placeholder, posture_chart_placeholder):
        # Emotion trend chart
        with emotion_chart_placeholder:
            if st.session_state.emotion_history:
                emotion_df = pd.DataFrame({
                    'Time': range(len(st.session_state.emotion_history)),
                    'Emotion': list(st.session_state.emotion_history)
                })
                
                fig = px.line(emotion_df, x='Time', y='Emotion', 
                             title="Emotion Timeline",
                             color_discrete_sequence=['#FF6B6B'])
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
        
        # Posture confidence chart
        with posture_chart_placeholder:
            if st.session_state.posture_history:
                posture_data = [p.get('confidence', 0) for p in st.session_state.posture_history]
                posture_df = pd.DataFrame({
                    'Time': range(len(posture_data)),
                    'Confidence': posture_data
                })
                
                fig = px.line(posture_df, x='Time', y='Confidence',
                             title="Posture Confidence",
                             color_discrete_sequence=['#4ECDC4'])
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

    def show_statistics(self):
        st.subheader("📊 Analysis Statistics")
        
        if not st.session_state.analysis_data:
            st.info("No analysis data available. Start recording to see statistics.")
            return
        
        # Create comprehensive statistics dashboard
        df = pd.DataFrame(st.session_state.analysis_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 😊 Emotion Statistics")
            self.visualizer.create_emotion_statistics(df)
        
        with col2:
            st.markdown("### 🤸 Posture Statistics")
            self.visualizer.create_posture_statistics(df)

    def get_emotion_color(self, emotion):
        colors = {
            'happy': '#FFD700',
            'sad': '#4169E1',
            'angry': '#FF4500',
            'fear': '#8B008B',
            'surprise': '#FF69B4',
            'disgust': '#32CD32',
            'neutral': '#808080'
        }
        return colors.get(emotion.lower(), '#808080')

    def run(self):
        settings = self.render_sidebar()
        self.render_main_interface(settings)

def main():
    app = EmoscanApp()
    app.run()

if __name__ == "__main__":
    main()
