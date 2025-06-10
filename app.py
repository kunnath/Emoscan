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
        if 'frame_count' not in st.session_state:
            st.session_state.frame_count = 0
        if 'camera_health' not in st.session_state:
            st.session_state.camera_health = 'unknown'
        if 'weapon_alerts' not in st.session_state:
            st.session_state.weapon_alerts = []

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
            weapon_detection = st.checkbox("🔫 Weapon Detection (Security)", True)
            save_data = st.checkbox("Save Analysis Data", False)
            
            # Weapon detection settings
            if weapon_detection:
                st.markdown("**Security Settings:**")
                weapon_sensitivity = st.slider("Weapon Detection Sensitivity", 0.1, 1.0, 0.3)
                alert_sound = st.checkbox("Alert Sound", True)
            else:
                weapon_sensitivity = 0.3
                alert_sound = False
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Camera status section
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.header("📹 Camera Status")
            
            if st.session_state.is_recording:
                st.success("🟢 Active")
                st.metric("Status", "Recording")
                if 'frame_count' in st.session_state:
                    st.metric("Frames Processed", st.session_state.get('frame_count', 0))
            else:
                st.error("🔴 Stopped")
                st.metric("Status", "Idle")
            
            # Connection health indicator
            if 'camera_health' in st.session_state:
                health = st.session_state.camera_health
                if health == 'good':
                    st.success("📡 Connection: Excellent")
                elif health == 'fair':
                    st.warning("📡 Connection: Fair")
                else:
                    st.error("📡 Connection: Poor")
            
            # Quick camera test
            if st.button("🔍 Test Camera Connection"):
                test_result = self.camera_handler.test_camera_connection(st.session_state.camera_source)
                if test_result:
                    st.success("✅ Camera connection successful!")
                else:
                    st.error("❌ Camera connection failed!")
            
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
            
            # Display weapon alerts if any
            if 'weapon_alerts' in st.session_state and st.session_state.weapon_alerts:
                st.markdown("### 🚨 Security Alerts")
                for alert in st.session_state.weapon_alerts[-3:]:  # Show last 3 alerts
                    alert_time = alert['timestamp'].strftime("%H:%M:%S")
                    if alert['alert_level'] == 'danger':
                        st.error(f"🚨 {alert_time}: {alert['message']}")
                    else:
                        st.warning(f"⚠️ {alert_time}: {alert['message']}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            return {
                'emotion_sensitivity': emotion_sensitivity,
                'posture_sensitivity': posture_sensitivity,
                'analyze_emotions': analyze_emotions,
                'analyze_posture': analyze_posture,
                'weapon_detection': weapon_detection,
                'weapon_sensitivity': weapon_sensitivity,
                'alert_sound': alert_sound,
                'save_data': save_data
            }

    def render_main_interface(self, settings):
        st.markdown('<h1 class="main-header">🎭 EmoScan - Emotion & Body Language Analyzer</h1>', 
                   unsafe_allow_html=True)
        
        # Control buttons
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            if st.button("▶️ Start Analysis", type="primary", disabled=st.session_state.is_recording):
                st.session_state.is_recording = True
                st.rerun()
                
        with col2:
            if st.button("⏹️ Stop Analysis", disabled=not st.session_state.is_recording):
                st.session_state.is_recording = False
                st.session_state.frame_count = 0
                st.session_state.camera_health = 'unknown'
                st.rerun()
                
        with col3:
            if st.button("🔄 Restart Camera"):
                if st.session_state.is_recording:
                    st.session_state.is_recording = False
                    time.sleep(0.5)
                st.session_state.is_recording = True
                st.rerun()
                
        with col4:
            if st.button("📊 Show Statistics"):
                self.show_statistics()
                
        with col5:
            if st.button("🗑️ Clear Data"):
                st.session_state.analysis_data = []
                st.session_state.emotion_history.clear()
                st.session_state.posture_history.clear()
                if 'weapon_alerts' in st.session_state:
                    st.session_state.weapon_alerts = []
                st.rerun()

        # Status indicator
        status_col1, status_col2 = st.columns(2)
        with status_col1:
            if st.session_state.is_recording:
                st.success("🟢 Camera Active - Recording in progress")
            else:
                st.info("🔴 Camera Stopped - Click 'Start Analysis' to begin")
        
        with status_col2:
            if st.session_state.is_recording:
                st.info(f"📷 Camera Source: {st.session_state.camera_source}")
            else:
                st.info(f"📷 Ready to connect to: {st.session_state.camera_source}")

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
        # Security alert display at the top if weapon detection is enabled
        if settings.get('weapon_detection', True):
            security_alert_placeholder = st.empty()
        
        # Create placeholder containers
        video_col, stats_col = st.columns([2, 1])
        
        with video_col:
            st.subheader("📹 Live Feed")
            video_placeholder = st.empty()
            
        with stats_col:
            st.subheader("📊 Real-time Analysis")
            emotion_placeholder = st.empty()
            posture_placeholder = st.empty()
            
            # Add weapon detection status
            if settings.get('weapon_detection', True):
                st.subheader("🔒 Security Status")
                weapon_status_placeholder = st.empty()
            
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
        consecutive_failures = 0
        max_consecutive_failures = 15  # Allow more failed reads before reconnecting
        reconnection_attempts = 0
        max_reconnection_attempts = 5
        
        # Create status placeholder for connection info
        connection_status_placeholder = st.empty()
        
        try:
            # Add a persistence check to prevent unexpected stops
            last_check_time = time.time()
            
            while st.session_state.is_recording:
                # Periodically verify that we should still be recording
                current_time = time.time()
                if current_time - last_check_time > 5:  # Check every 5 seconds
                    # Verify session state hasn't been corrupted
                    if not hasattr(st.session_state, 'is_recording') or not st.session_state.is_recording:
                        st.warning("⚠️ Recording state lost. Camera will attempt to continue...")
                        st.session_state.is_recording = True
                    last_check_time = current_time
                # Check if camera is still valid
                if cap is None or not cap.isOpened():
                    st.warning("🔄 Camera disconnected. Attempting to reconnect...")
                    reconnection_attempts += 1
                    
                    if reconnection_attempts > max_reconnection_attempts:
                        st.error(f"❌ Failed to reconnect after {max_reconnection_attempts} attempts. Stopping camera.")
                        break
                    
                    # Try to reinitialize camera
                    cap = self.camera_handler.initialize_camera(st.session_state.camera_source)
                    if cap is None:
                        st.error(f"❌ Reconnection attempt {reconnection_attempts}/{max_reconnection_attempts} failed. Retrying in 3 seconds...")
                        time.sleep(3)
                        continue
                    else:
                        st.success("✅ Camera reconnected successfully!")
                        consecutive_failures = 0
                        reconnection_attempts = 0
                
                try:
                    ret, frame = cap.read()
                    
                    if not ret or frame is None:
                        consecutive_failures += 1
                        
                        # Show warning but don't stop immediately
                        if consecutive_failures <= 3:
                            connection_status_placeholder.warning(f"⚠️ Camera read failed (attempt {consecutive_failures}/{max_consecutive_failures})")
                        elif consecutive_failures == max_consecutive_failures:
                            connection_status_placeholder.warning("🔄 Too many failures. Attempting to reconnect...")
                        
                        # Try to reconnect if too many consecutive failures
                        if consecutive_failures >= max_consecutive_failures:
                            cap.release()
                            time.sleep(1)  # Brief wait before reconnecting
                            
                            # Try to reinitialize camera
                            cap = self.camera_handler.initialize_camera(st.session_state.camera_source)
                            if cap is None:
                                consecutive_failures = 0  # Reset to avoid infinite failures
                                time.sleep(2)  # Wait longer before next attempt
                                continue
                            else:
                                connection_status_placeholder.success("✅ Camera reconnected!")
                                consecutive_failures = 0
                        
                        time.sleep(0.2)  # Brief pause before next attempt
                        continue
                    
                    # Clear connection status on successful read
                    if consecutive_failures > 0:
                        connection_status_placeholder.empty()
                    
                    # Reset failure counter on successful read
                    consecutive_failures = 0
                    frame_count += 1
                    st.session_state.frame_count = frame_count
                    
                    # Update camera health based on success rate
                    if frame_count % 30 == 0:  # Check health every 30 frames
                        if consecutive_failures == 0:
                            st.session_state.camera_health = 'good'
                        elif consecutive_failures < 3:
                            st.session_state.camera_health = 'fair'
                        else:
                            st.session_state.camera_health = 'poor'
                    
                    # Process frame every few frames to improve performance
                    if frame_count % 2 == 0:  # Process more frequently for better responsiveness
                        try:
                            analysis_result = self.analyze_frame(frame, settings)
                            
                            # Update displays
                            self.update_video_display(video_placeholder, frame, analysis_result)
                            self.update_stats_display(emotion_placeholder, posture_placeholder, analysis_result)
                            
                            # Update weapon detection status
                            if settings.get('weapon_detection', True):
                                self.update_weapon_status(weapon_status_placeholder, analysis_result)
                                
                                # Update security alert at top if there's an active threat
                                if ('posture' in analysis_result and 
                                    analysis_result['posture'] and 
                                    analysis_result['posture'].get('alert_active', False)):
                                    self.update_security_alert(security_alert_placeholder, analysis_result)
                                else:
                                    security_alert_placeholder.empty()
                            
                            # Update charts less frequently to improve performance
                            if frame_count % 20 == 0:
                                self.update_charts(emotion_chart_placeholder, posture_chart_placeholder)
                        
                        except Exception as analysis_error:
                            st.warning(f"⚠️ Analysis error (continuing): {str(analysis_error)}")
                            # Continue processing even if analysis fails
                            continue
                    
                    # Smaller delay to improve responsiveness
                    time.sleep(0.05)
                
                except Exception as frame_error:
                    consecutive_failures += 1
                    st.warning(f"⚠️ Frame processing error: {str(frame_error)}")
                    
                    # If too many frame errors, suggest recovery actions
                    if consecutive_failures > 5:
                        st.error("🚨 Multiple frame processing errors detected!")
                        st.info("""
                        **Recovery suggestions:**
                        - Click 'Restart Camera' button
                        - Check your camera connection
                        - Try reducing video quality in camera settings
                        - Close other applications using the camera
                        """)
                    
                    time.sleep(0.1)
                    continue
                
        except Exception as e:
            st.error(f"❌ Critical error during analysis: {str(e)}")
            st.info("💡 Try clicking 'Stop Analysis' and then 'Start Analysis' again")
        finally:
            # Clean up camera resources
            if cap is not None:
                try:
                    cap.release()
                    st.info("📷 Camera stopped successfully")
                except Exception as release_error:
                    st.warning(f"⚠️ Error releasing camera: {str(release_error)}")

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
                
                # Check for weapon alerts
                if posture_result and posture_result.get('alert_active', False):
                    weapon_detection = posture_result.get('weapon_detection', {})
                    
                    # Display weapon alert
                    if weapon_detection.get('weapons_detected', False):
                        alert_level = weapon_detection.get('alert_level', 'warning')
                        alert_message = weapon_detection.get('alert_message', 'Weapon detected!')
                        
                        # Store alert in session state
                        if 'weapon_alerts' not in st.session_state:
                            st.session_state.weapon_alerts = []
                        
                        st.session_state.weapon_alerts.append({
                            'timestamp': timestamp,
                            'alert_level': alert_level,
                            'message': alert_message,
                            'weapon_count': weapon_detection.get('weapon_count', 0),
                            'weapon_in_hands': posture_result.get('weapon_in_hands', False)
                        })
                        
                        # Keep only recent alerts (last 10)
                        st.session_state.weapon_alerts = st.session_state.weapon_alerts[-10:]
                        
                        # Show immediate alert in UI
                        if alert_level == 'danger':
                            st.error(f"🚨 SECURITY ALERT: {alert_message}")
                        elif alert_level == 'warning':
                            st.warning(f"⚠️ SECURITY WARNING: {alert_message}")
            
            # Save data if enabled
            if settings['save_data']:
                analysis_data = {
                    'timestamp': timestamp,
                    'emotions': results.get('emotions', {}),
                    'posture': results.get('posture', {}),
                    'weapon_detection': results.get('posture', {}).get('weapon_detection', {}) if 'posture' in results else {}
                }
                st.session_state.analysis_data.append(analysis_data)
            
        except Exception as e:
            st.error(f"Analysis error: {str(e)}")
            
        return results

    def update_video_display(self, placeholder, frame, analysis_result):
        # Draw analysis overlays on frame
        annotated_frame = frame.copy()
        
        # Use processed frame from weapon detection if available
        if ('posture' in analysis_result and 
            analysis_result['posture'] and 
            'processed_frame' in analysis_result['posture']):
            annotated_frame = analysis_result['posture']['processed_frame'].copy()
        
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
        
        # Add weapon detection status overlay
        if ('posture' in analysis_result and 
            analysis_result['posture'] and 
            'weapon_detection' in analysis_result['posture']):
            weapon_detection = analysis_result['posture']['weapon_detection']
            
            # Add status text overlay
            status_color = self._get_alert_color(weapon_detection.get('alert_level', 'safe'))
            status_text = weapon_detection.get('alert_message', '✅ No weapons detected')
            
            # Draw status box
            cv2.rectangle(annotated_frame, (10, 10), (400, 60), (0, 0, 0), -1)
            cv2.rectangle(annotated_frame, (10, 10), (400, 60), status_color, 2)
            cv2.putText(annotated_frame, status_text, (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Convert to RGB for Streamlit
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
    
    def _get_alert_color(self, alert_level):
        """Get BGR color for alert level"""
        colors = {
            'safe': (0, 255, 0),      # Green
            'warning': (0, 165, 255),  # Orange
            'danger': (0, 0, 255)      # Red
        }
        return colors.get(alert_level, (255, 255, 255))
        
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

    def update_weapon_status(self, placeholder, analysis_result):
        """Update weapon detection status display."""
        if 'posture' not in analysis_result or not analysis_result['posture']:
            return
            
        posture_data = analysis_result['posture']
        weapon_detected = posture_data.get('weapon_detected', False)
        weapon_confidence = posture_data.get('weapon_confidence', 0)
        weapon_in_hand = posture_data.get('weapon_in_hand', False)
        weapon_type = posture_data.get('weapon_type', 'Unknown')
        
        if weapon_detected:
            # Store weapon alert for history
            alert_data = {
                'timestamp': datetime.now(),
                'weapon_type': weapon_type,
                'confidence': weapon_confidence,
                'in_hand': weapon_in_hand
            }
            st.session_state.weapon_alerts.append(alert_data)
            
            # Keep only last 20 alerts
            if len(st.session_state.weapon_alerts) > 20:
                st.session_state.weapon_alerts.pop(0)
            
            # Display current weapon status
            status_color = "🔴" if weapon_in_hand else "🟡"
            status_text = "IN HAND" if weapon_in_hand else "DETECTED"
            
            with placeholder.container():
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%); 
                           padding: 1rem; border-radius: 10px; color: white; 
                           text-align: center; margin: 0.5rem 0;">
                    <h3 style="margin: 0; color: white;">{status_color} WEAPON {status_text}</h3>
                    <p style="margin: 0.5rem 0; color: white;"><strong>Type:</strong> {weapon_type}</p>
                    <p style="margin: 0; color: white;"><strong>Confidence:</strong> {weapon_confidence:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show recent alerts
                if st.session_state.weapon_alerts:
                    st.markdown("**Recent Weapon Alerts:**")
                    for i, alert in enumerate(reversed(st.session_state.weapon_alerts[-5:])):
                        status_icon = "🔴" if alert['in_hand'] else "🟡"
                        time_str = alert['timestamp'].strftime("%H:%M:%S")
                        st.markdown(f"{status_icon} {time_str} - {alert['weapon_type']} ({alert['confidence']:.1%})")
        else:
            with placeholder.container():
                st.markdown("""
                <div style="background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%); 
                           padding: 1rem; border-radius: 10px; color: white; 
                           text-align: center; margin: 0.5rem 0;">
                    <h3 style="margin: 0; color: white;">✅ NO WEAPONS DETECTED</h3>
                    <p style="margin: 0; color: white;">Area secure</p>
                </div>
                """, unsafe_allow_html=True)

    def update_security_alert(self, placeholder, analysis_result):
        """Update security alert display at the top of the page."""
        if 'posture' not in analysis_result or not analysis_result['posture']:
            return
            
        posture_data = analysis_result['posture']
        
        if posture_data.get('alert_active', False):
            weapon_type = posture_data.get('weapon_type', 'Unknown')
            weapon_confidence = posture_data.get('weapon_confidence', 0)
            weapon_in_hand = posture_data.get('weapon_in_hand', False)
            
            alert_level = "CRITICAL" if weapon_in_hand else "WARNING"
            alert_color = "#ff0000" if weapon_in_hand else "#ff8800"
            alert_icon = "🚨" if weapon_in_hand else "⚠️"
            
            with placeholder.container():
                st.markdown(f"""
                <div style="background: {alert_color}; padding: 1rem; border-radius: 10px; 
                           color: white; text-align: center; margin: 1rem 0; 
                           border: 3px solid #ffffff; box-shadow: 0 0 20px rgba(255,0,0,0.5);
                           animation: pulse 1s infinite;">
                    <h1 style="margin: 0; color: white; font-size: 2rem;">
                        {alert_icon} SECURITY ALERT - {alert_level} {alert_icon}
                    </h1>
                    <h2 style="margin: 0.5rem 0; color: white;">
                        WEAPON DETECTED: {weapon_type.upper()}
                    </h2>
                    <p style="margin: 0; color: white; font-size: 1.2rem;">
                        Status: {'WEAPON IN HAND' if weapon_in_hand else 'WEAPON VISIBLE'} 
                        | Confidence: {weapon_confidence:.1%}
                    </p>
                </div>
                <style>
                @keyframes pulse {{
                    0% {{ opacity: 1; }}
                    50% {{ opacity: 0.7; }}
                    100% {{ opacity: 1; }}
                }}
                </style>
                """, unsafe_allow_html=True)

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
