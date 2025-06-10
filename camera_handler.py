import cv2
import numpy as np
import logging
import time

class CameraHandler:
    def __init__(self):
        self.cap = None
        self.camera_type = None
        
    def initialize_camera(self, camera_source):
        """
        Initialize camera based on source type
        
        Args:
            camera_source: Camera source (webcam, IP camera URL, etc.)
            
        Returns:
            OpenCV VideoCapture object or None if failed
        """
        try:
            if camera_source == "webcam":
                return self._initialize_webcam()
            elif camera_source.startswith("http"):
                return self._initialize_ip_camera(camera_source)
            else:
                # Try to parse as camera index
                try:
                    camera_index = int(camera_source)
                    return self._initialize_webcam(camera_index)
                except ValueError:
                    logging.error(f"Invalid camera source: {camera_source}")
                    return None
                    
        except Exception as e:
            logging.error(f"Error initializing camera: {str(e)}")
            return None
    
    def _initialize_webcam(self, camera_index=0):
        """
        Initialize local webcam with improved error handling
        """
        try:
            # Try different backends if default fails
            backends = [cv2.CAP_ANY, cv2.CAP_DSHOW, cv2.CAP_V4L2]
            
            for backend in backends:
                try:
                    cap = cv2.VideoCapture(camera_index, backend)
                    
                    if not cap.isOpened():
                        cap.release()
                        continue
                    
                    # Set camera properties for better performance
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size
                    
                    # Test if camera works with multiple attempts
                    for test_attempt in range(3):
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            self.camera_type = "webcam"
                            logging.info(f"Successfully initialized webcam {camera_index} with backend {backend}")
                            return cap
                        time.sleep(0.1)
                    
                    cap.release()
                    
                except Exception as e:
                    logging.warning(f"Failed to initialize webcam with backend {backend}: {str(e)}")
                    continue
            
            logging.error(f"Failed to open webcam {camera_index} with any backend")
            return None
            
        except Exception as e:
            logging.error(f"Error initializing webcam: {str(e)}")
            return None
    
    def _initialize_ip_camera(self, url):
        """
        Initialize IP/WiFi camera with improved reliability
        """
        try:
            # Add timeout and buffer settings for IP cameras
            cap = cv2.VideoCapture(url)
            
            if not cap.isOpened():
                logging.error(f"Failed to open IP camera: {url}")
                return None
            
            # Set properties for IP camera
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize delay
            cap.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS for network cameras
            
            # For IP cameras, try to set additional properties that might help
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            except:
                pass  # Ignore if not supported
            
            # Test connection with multiple attempts and longer timeout
            start_time = time.time()
            timeout = 15  # 15 seconds timeout
            successful_reads = 0
            required_reads = 3  # Require multiple successful reads
            
            while time.time() - start_time < timeout and successful_reads < required_reads:
                try:
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        successful_reads += 1
                        if successful_reads >= required_reads:
                            self.camera_type = "ip_camera"
                            logging.info(f"Successfully connected to IP camera: {url}")
                            return cap
                    else:
                        # Clear buffer if read fails
                        for _ in range(5):
                            cap.read()
                    
                    time.sleep(0.2)
                except Exception as read_error:
                    logging.warning(f"IP camera read attempt failed: {str(read_error)}")
                    time.sleep(0.5)
            
            logging.error(f"Timeout or insufficient successful reads for IP camera: {url}")
            cap.release()
            return None
            
        except Exception as e:
            logging.error(f"Error connecting to IP camera {url}: {str(e)}")
            return None
            
        except Exception as e:
            logging.error(f"Error connecting to IP camera {url}: {str(e)}")
            return None
    
    def test_camera_connection(self, camera_source):
        """
        Test camera connection without keeping it open
        
        Args:
            camera_source: Camera source to test
            
        Returns:
            Boolean indicating if connection is successful
        """
        cap = self.initialize_camera(camera_source)
        if cap is not None:
            cap.release()
            return True
        return False
    
    def get_camera_info(self, cap):
        """
        Get camera information and properties
        
        Args:
            cap: OpenCV VideoCapture object
            
        Returns:
            Dictionary with camera information
        """
        if cap is None or not cap.isOpened():
            return None
        
        try:
            info = {
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'backend': cap.getBackendName(),
                'camera_type': self.camera_type
            }
            
            # Try to get additional properties (may not work for all cameras)
            try:
                info['brightness'] = cap.get(cv2.CAP_PROP_BRIGHTNESS)
                info['contrast'] = cap.get(cv2.CAP_PROP_CONTRAST)
                info['saturation'] = cap.get(cv2.CAP_PROP_SATURATION)
            except:
                pass
            
            return info
            
        except Exception as e:
            logging.error(f"Error getting camera info: {str(e)}")
            return None
    
    def optimize_camera_settings(self, cap):
        """
        Optimize camera settings for emotion and pose detection
        
        Args:
            cap: OpenCV VideoCapture object
        """
        if cap is None or not cap.isOpened():
            return False
        
        try:
            # Set optimal resolution for processing speed vs quality balance
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Set frame rate
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Optimize for face detection (increase brightness/contrast if possible)
            try:
                cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
                cap.set(cv2.CAP_PROP_CONTRAST, 0.5)
                cap.set(cv2.CAP_PROP_SATURATION, 0.5)
            except:
                # These properties might not be available for all cameras
                pass
            
            # Set auto-exposure for consistent lighting
            try:
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
            except:
                pass
            
            logging.info("Camera settings optimized")
            return True
            
        except Exception as e:
            logging.error(f"Error optimizing camera settings: {str(e)}")
            return False
    
    def capture_frame_with_retry(self, cap, max_retries=5):
        """
        Capture frame with retry mechanism for unreliable connections
        
        Args:
            cap: OpenCV VideoCapture object
            max_retries: Maximum number of retry attempts
            
        Returns:
            Tuple of (success, frame)
        """
        if cap is None or not cap.isOpened():
            return False, None
        
        for attempt in range(max_retries):
            try:
                ret, frame = cap.read()
                
                if ret and frame is not None and frame.size > 0:
                    return True, frame
                
                # If first attempt fails, try to flush buffer for IP cameras
                if attempt == 0 and self.camera_type == "ip_camera":
                    # Try to read and discard a few frames to clear buffer
                    for _ in range(3):
                        cap.read()
                    time.sleep(0.1)  # Brief pause before retry
                    
            except Exception as e:
                logging.warning(f"Frame capture attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(0.2 * (attempt + 1))  # Increasing delay between retries
        
        return False, None
    
    def is_camera_healthy(self, cap):
        """
        Check if camera connection is healthy
        
        Args:
            cap: OpenCV VideoCapture object
            
        Returns:
            Boolean indicating camera health
        """
        if cap is None:
            return False
        
        try:
            return cap.isOpened()
        except Exception:
            return False
    
    def reconnect_camera(self, camera_source, max_attempts=3):
        """
        Attempt to reconnect to camera with multiple tries
        
        Args:
            camera_source: Camera source to reconnect to
            max_attempts: Maximum reconnection attempts
            
        Returns:
            OpenCV VideoCapture object or None if failed
        """
        for attempt in range(max_attempts):
            try:
                logging.info(f"Reconnection attempt {attempt + 1}/{max_attempts}")
                
                # Wait progressively longer between attempts
                if attempt > 0:
                    wait_time = attempt * 2
                    logging.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                
                cap = self.initialize_camera(camera_source)
                if cap is not None:
                    logging.info(f"Successfully reconnected on attempt {attempt + 1}")
                    return cap
                    
            except Exception as e:
                logging.error(f"Reconnection attempt {attempt + 1} failed: {str(e)}")
        
        logging.error(f"Failed to reconnect after {max_attempts} attempts")
        return None
    
    def preprocess_frame(self, frame):
        """
        Preprocess frame for better analysis results
        
        Args:
            frame: Input frame
            
        Returns:
            Preprocessed frame
        """
        if frame is None:
            return None
        
        try:
            # Resize if frame is too large (for performance)
            height, width = frame.shape[:2]
            max_width = 1280
            max_height = 720
            
            if width > max_width or height > max_height:
                scale = min(max_width / width, max_height / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Enhance contrast and brightness for better face detection
            frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=10)
            
            # Reduce noise
            frame = cv2.bilateralFilter(frame, 9, 75, 75)
            
            return frame
            
        except Exception as e:
            logging.error(f"Error preprocessing frame: {str(e)}")
            return frame
    
    def create_camera_config_guide(self):
        """
        Return configuration guide for different camera types
        """
        guide = {
            'webcam': {
                'description': 'Built-in or USB webcam',
                'setup': 'Simply select "Webcam" option',
                'troubleshooting': [
                    'Make sure webcam is not being used by another application',
                    'Check webcam permissions in system settings',
                    'Try different USB ports for external webcams'
                ]
            },
            'ip_camera': {
                'description': 'Network IP camera with MJPEG/RTSP stream',
                'setup': [
                    'Ensure camera and computer are on same network',
                    'Find camera IP address (usually in camera settings)',
                    'Use format: http://IP:PORT/video or http://IP:PORT/stream'
                ],
                'common_urls': [
                    'http://192.168.1.100:8080/video',
                    'rtsp://192.168.1.100:554/stream',
                    'http://admin:password@192.168.1.100/mjpeg'
                ],
                'troubleshooting': [
                    'Check if camera stream works in web browser first',
                    'Verify network connectivity',
                    'Check camera authentication requirements',
                    'Try different stream formats (MJPEG vs RTSP)'
                ]
            },
            'phone_camera': {
                'description': 'Smartphone camera via IP Webcam app',
                'setup': [
                    'Install "IP Webcam" app on Android phone',
                    'Connect phone to same WiFi network',
                    'Start server in IP Webcam app',
                    'Use the displayed URL (usually ending with /video)'
                ],
                'ios_alternatives': [
                    'EpocCam',
                    'iVCam',
                    'DroidCam (also available for Android)'
                ],
                'troubleshooting': [
                    'Ensure phone and computer are on same WiFi',
                    'Check firewall settings',
                    'Try different video quality settings in app',
                    'Keep phone charged and prevent screen timeout'
                ]
            }
        }
        return guide
    
    def release_camera(self, cap):
        """
        Properly release camera resources
        """
        if cap is not None:
            try:
                cap.release()
                logging.info("Camera released successfully")
            except Exception as e:
                logging.error(f"Error releasing camera: {str(e)}")
    
    def __del__(self):
        """
        Cleanup when object is destroyed
        """
        if self.cap is not None:
            self.release_camera(self.cap)
