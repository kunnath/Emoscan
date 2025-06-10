import cv2
import numpy as np
import logging
import urllib.request
import os
from typing import List, Tuple, Dict, Optional

class WeaponDetector:
    """
    Real-time weapon detection using YOLO object detection.
    Detects various types of weapons including guns, knives, and other dangerous objects.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Weapon class names that we want to detect
        self.weapon_classes = [
            'knife', 'gun', 'pistol', 'rifle', 'handgun', 'weapon', 
            'sword', 'blade', 'machete', 'dagger', 'revolver',
            'firearm', 'shotgun', 'carbine'
        ]
        
        # Alert configuration
        self.alert_threshold = 0.3  # Confidence threshold for weapon detection
        self.alert_active = False
        self.last_detection_time = 0
        
        # Initialize YOLO detector
        self.net = None
        self.output_layers = None
        self.classes = None
        self.colors = None
        
        # Try to load YOLO model
        self._initialize_yolo()
    
    def _initialize_yolo(self):
        """Initialize YOLO model for object detection"""
        try:
            # Check if YOLO files exist, if not download a lightweight model
            weights_path = "yolo_models/yolov4-tiny.weights"
            config_path = "yolo_models/yolov4-tiny.cfg"
            names_path = "yolo_models/coco.names"
            
            # Create directory if it doesn't exist
            os.makedirs("yolo_models", exist_ok=True)
            
            # If files don't exist, we'll use a simple contour-based approach
            if not all(os.path.exists(path) for path in [weights_path, config_path, names_path]):
                self.logger.warning("YOLO model files not found. Using alternative detection method.")
                self._initialize_alternative_detector()
                return
            
            # Load YOLO
            self.net = cv2.dnn.readNet(weights_path, config_path)
            layer_names = self.net.getLayerNames()
            self.output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
            
            # Load class names
            with open(names_path, "r") as f:
                self.classes = [line.strip() for line in f.readlines()]
            
            # Generate colors for each class
            self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
            
            self.logger.info("YOLO weapon detector initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize YOLO: {e}")
            self._initialize_alternative_detector()
    
    def _initialize_alternative_detector(self):
        """Initialize alternative weapon detection using shape analysis"""
        self.logger.info("Initializing alternative weapon detection using shape analysis")
        # This will use contour and shape analysis for basic weapon detection
        self.use_alternative = True
    
    def detect_weapons(self, frame: np.ndarray) -> Dict:
        """
        Detect weapons in the given frame
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            Dictionary containing detection results and alert status
        """
        try:
            height, width, channels = frame.shape
            
            # Initialize result structure
            result = {
                'weapons_detected': False,
                'weapon_count': 0,
                'detections': [],
                'alert_level': 'safe',  # safe, warning, danger
                'alert_message': '',
                'processed_frame': frame.copy()
            }
            
            if self.net is not None:
                # Use YOLO detection
                result = self._detect_with_yolo(frame, result)
            else:
                # Use alternative detection method
                result = self._detect_with_alternative(frame, result)
            
            # Determine alert level
            result = self._determine_alert_level(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in weapon detection: {e}")
            return {
                'weapons_detected': False,
                'weapon_count': 0,
                'detections': [],
                'alert_level': 'safe',
                'alert_message': f'Detection error: {str(e)}',
                'processed_frame': frame
            }
    
    def _detect_with_yolo(self, frame: np.ndarray, result: Dict) -> Dict:
        """Detect weapons using YOLO model"""
        height, width, channels = frame.shape
        
        # Prepare image for YOLO
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        
        # Process detections
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.alert_threshold:
                    class_name = self.classes[class_id].lower()
                    
                    # Check if detected object is a weapon
                    if any(weapon in class_name for weapon in self.weapon_classes):
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
        
        # Apply non-maximum suppression
        if boxes:
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, self.alert_threshold, 0.4)
            
            if len(indexes) > 0:
                result['weapons_detected'] = True
                result['weapon_count'] = len(indexes.flatten())
                
                for i in indexes.flatten():
                    x, y, w, h = boxes[i]
                    label = f"{self.classes[class_ids[i]]}: {confidences[i]:.2f}"
                    color = self.colors[class_ids[i]]
                    
                    # Draw bounding box and label
                    cv2.rectangle(result['processed_frame'], (x, y), (x + w, y + h), color, 2)
                    cv2.putText(result['processed_frame'], label, (x, y - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Add to detections list
                    result['detections'].append({
                        'class': self.classes[class_ids[i]],
                        'confidence': confidences[i],
                        'bbox': [x, y, w, h],
                        'center': [x + w//2, y + h//2]
                    })
        
        return result
    
    def _detect_with_alternative(self, frame: np.ndarray, result: Dict) -> Dict:
        """Alternative weapon detection using shape and color analysis"""
        # Convert to different color spaces for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Look for sharp metallic objects (potential knives/blades)
        weapon_detected = self._detect_sharp_objects(gray, result)
        
        # Look for gun-like shapes
        weapon_detected |= self._detect_gun_shapes(gray, result)
        
        # Look for metallic reflections
        weapon_detected |= self._detect_metallic_surfaces(hsv, result)
        
        if weapon_detected:
            result['weapons_detected'] = True
            result['weapon_count'] = len(result['detections'])
        
        return result
    
    def _detect_sharp_objects(self, gray: np.ndarray, result: Dict) -> bool:
        """Detect sharp objects that could be weapons"""
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        weapon_found = False
        
        for contour in contours:
            # Filter by area
            area = cv2.contourArea(contour)
            if area < 500 or area > 50000:
                continue
            
            # Calculate aspect ratio
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            
            # Look for elongated objects (potential knives)
            if aspect_ratio > 3 or aspect_ratio < 0.3:
                # Calculate solidity (filled area vs convex hull area)
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / hull_area if hull_area > 0 else 0
                
                # Sharp objects typically have lower solidity
                if solidity < 0.7:
                    weapon_found = True
                    
                    # Draw detection on frame
                    cv2.rectangle(result['processed_frame'], (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(result['processed_frame'], "POTENTIAL WEAPON", (x, y - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    result['detections'].append({
                        'class': 'potential_sharp_object',
                        'confidence': 0.6,
                        'bbox': [x, y, w, h],
                        'center': [x + w//2, y + h//2]
                    })
        
        return weapon_found
    
    def _detect_gun_shapes(self, gray: np.ndarray, result: Dict) -> bool:
        """Detect gun-like shapes"""
        # This is a simplified approach - in practice, you'd want more sophisticated methods
        # Look for L-shaped or rectangular objects with specific proportions
        
        # Apply morphological operations
        kernel = np.ones((3, 3), np.uint8)
        processed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        weapon_found = False
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1000 or area > 30000:
                continue
            
            # Approximate contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Look for rectangular shapes that could be guns
            if len(approx) >= 4:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                
                # Guns typically have specific aspect ratios
                if 1.5 < aspect_ratio < 4:
                    weapon_found = True
                    
                    cv2.rectangle(result['processed_frame'], (x, y), (x + w, y + h), (0, 165, 255), 2)
                    cv2.putText(result['processed_frame'], "POTENTIAL FIREARM", (x, y - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                    
                    result['detections'].append({
                        'class': 'potential_firearm',
                        'confidence': 0.5,
                        'bbox': [x, y, w, h],
                        'center': [x + w//2, y + h//2]
                    })
        
        return weapon_found
    
    def _detect_metallic_surfaces(self, hsv: np.ndarray, result: Dict) -> bool:
        """Detect metallic surfaces that could indicate weapons"""
        # Define metallic color ranges (grayish, metallic)
        lower_metallic = np.array([0, 0, 50])
        upper_metallic = np.array([180, 50, 255])
        
        # Create mask for metallic colors
        mask = cv2.inRange(hsv, lower_metallic, upper_metallic)
        
        # Find contours in mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        weapon_found = False
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 800:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # If metallic surface is found in hand region, increase suspicion
            # This would need integration with pose detection to identify hand positions
            weapon_found = True
            
            cv2.rectangle(result['processed_frame'], (x, y), (x + w, y + h), (255, 255, 0), 2)
            cv2.putText(result['processed_frame'], "METALLIC OBJECT", (x, y - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            result['detections'].append({
                'class': 'metallic_object',
                'confidence': 0.4,
                'bbox': [x, y, w, h],
                'center': [x + w//2, y + h//2]
            })
        
        return weapon_found
    
    def _determine_alert_level(self, result: Dict) -> Dict:
        """Determine the alert level based on detections"""
        if not result['weapons_detected']:
            result['alert_level'] = 'safe'
            result['alert_message'] = '✅ No weapons detected'
        elif result['weapon_count'] == 1:
            result['alert_level'] = 'warning'
            result['alert_message'] = '⚠️ WARNING: Potential weapon detected!'
        else:
            result['alert_level'] = 'danger'
            result['alert_message'] = f'🚨 DANGER: {result["weapon_count"]} potential weapons detected!'
        
        return result
    
    def is_weapon_in_hand_region(self, detection: Dict, hand_landmarks) -> bool:
        """
        Check if detected weapon is in hand region
        This requires integration with MediaPipe hand detection
        """
        # This method would be implemented to check if weapon detection
        # overlaps with hand landmark positions
        return True  # Placeholder implementation
    
    def get_alert_color(self, alert_level: str) -> Tuple[int, int, int]:
        """Get color for alert level"""
        colors = {
            'safe': (0, 255, 0),      # Green
            'warning': (0, 165, 255),  # Orange
            'danger': (0, 0, 255)      # Red
        }
        return colors.get(alert_level, (255, 255, 255))
