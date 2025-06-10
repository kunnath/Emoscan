import cv2
import numpy as np
import mediapipe as mp
import math
import logging

class BodyLanguageAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Define pose classifications
        self.posture_types = {
            'upright': 'Good Posture',
            'slouched': 'Slouched',
            'leaning_left': 'Leaning Left',
            'leaning_right': 'Leaning Right',
            'head_down': 'Head Down',
            'arms_crossed': 'Arms Crossed',
            'hands_on_hips': 'Hands on Hips',
            'relaxed': 'Relaxed'
        }
    
    def analyze_posture(self, frame, sensitivity=0.6):
        """
        Analyze body language and posture from the frame
        
        Args:
            frame: Input image frame
            sensitivity: Confidence threshold for pose detection
            
        Returns:
            Dictionary containing posture analysis results
        """
        try:
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = self.pose.process(rgb_frame)
            
            if not results.pose_landmarks:
                return None
            
            # Extract landmark coordinates
            landmarks = self._extract_landmarks(results.pose_landmarks, frame.shape)
            
            if not landmarks:
                return None
            
            # Analyze posture
            posture_analysis = self._analyze_pose_geometry(landmarks)
            posture_type = self._classify_posture(posture_analysis)
            
            # Calculate overall confidence
            confidence = self._calculate_pose_confidence(results.pose_landmarks)
            
            if confidence < sensitivity:
                return None
            
            return {
                'posture_type': posture_type,
                'confidence': confidence,
                'landmarks': landmarks,
                'pose_analysis': posture_analysis,
                'keypoints_detected': len(landmarks),
                'pose_landmarks': results.pose_landmarks,
                'body_detected': True
            }
            
        except Exception as e:
            logging.error(f"Error in posture analysis: {str(e)}")
            return None
    
    def _extract_landmarks(self, pose_landmarks, frame_shape):
        """
        Extract key pose landmarks with normalized coordinates
        """
        height, width = frame_shape[:2]
        landmarks = {}
        
        # Key landmarks for posture analysis
        key_points = {
            'nose': self.mp_pose.PoseLandmark.NOSE,
            'left_shoulder': self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            'right_shoulder': self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            'left_elbow': self.mp_pose.PoseLandmark.LEFT_ELBOW,
            'right_elbow': self.mp_pose.PoseLandmark.RIGHT_ELBOW,
            'left_wrist': self.mp_pose.PoseLandmark.LEFT_WRIST,
            'right_wrist': self.mp_pose.PoseLandmark.RIGHT_WRIST,
            'left_hip': self.mp_pose.PoseLandmark.LEFT_HIP,
            'right_hip': self.mp_pose.PoseLandmark.RIGHT_HIP,
            'left_knee': self.mp_pose.PoseLandmark.LEFT_KNEE,
            'right_knee': self.mp_pose.PoseLandmark.RIGHT_KNEE
        }
        
        for name, landmark_id in key_points.items():
            landmark = pose_landmarks.landmark[landmark_id]
            if landmark.visibility > 0.5:  # Only include visible landmarks
                landmarks[name] = {
                    'x': int(landmark.x * width),
                    'y': int(landmark.y * height),
                    'z': landmark.z,
                    'visibility': landmark.visibility
                }
        
        return landmarks
    
    def _analyze_pose_geometry(self, landmarks):
        """
        Analyze geometric relationships between body parts
        """
        analysis = {}
        
        try:
            # Shoulder alignment
            if 'left_shoulder' in landmarks and 'right_shoulder' in landmarks:
                left_shoulder = landmarks['left_shoulder']
                right_shoulder = landmarks['right_shoulder']
                
                # Calculate shoulder angle
                shoulder_angle = self._calculate_angle_horizontal(left_shoulder, right_shoulder)
                analysis['shoulder_tilt'] = shoulder_angle
                
                # Shoulder center
                shoulder_center_x = (left_shoulder['x'] + right_shoulder['x']) / 2
                shoulder_center_y = (left_shoulder['y'] + right_shoulder['y']) / 2
                analysis['shoulder_center'] = {'x': shoulder_center_x, 'y': shoulder_center_y}
            
            # Head position relative to shoulders
            if 'nose' in landmarks and 'shoulder_center' in analysis:
                nose = landmarks['nose']
                shoulder_center = analysis['shoulder_center']
                
                # Head alignment
                head_offset_x = nose['x'] - shoulder_center['x']
                head_offset_y = nose['y'] - shoulder_center['y']
                
                analysis['head_offset_x'] = head_offset_x
                analysis['head_offset_y'] = head_offset_y
                analysis['head_forward'] = head_offset_y > 50  # Head protruding forward
            
            # Arm positions
            self._analyze_arm_positions(landmarks, analysis)
            
            # Hip alignment
            if 'left_hip' in landmarks and 'right_hip' in landmarks:
                left_hip = landmarks['left_hip']
                right_hip = landmarks['right_hip']
                hip_angle = self._calculate_angle_horizontal(left_hip, right_hip)
                analysis['hip_tilt'] = hip_angle
            
            # Overall posture score
            analysis['posture_score'] = self._calculate_posture_score(analysis)
            
        except Exception as e:
            logging.error(f"Error in pose geometry analysis: {str(e)}")
        
        return analysis
    
    def _analyze_arm_positions(self, landmarks, analysis):
        """
        Analyze arm positions and gestures
        """
        # Check for arms crossed
        if all(k in landmarks for k in ['left_wrist', 'right_wrist', 'left_shoulder', 'right_shoulder']):
            left_wrist = landmarks['left_wrist']
            right_wrist = landmarks['right_wrist']
            left_shoulder = landmarks['left_shoulder']
            right_shoulder = landmarks['right_shoulder']
            
            # Arms crossed detection
            arms_crossed = (left_wrist['x'] > right_shoulder['x'] and 
                          right_wrist['x'] < left_shoulder['x'])
            analysis['arms_crossed'] = arms_crossed
        
        # Check for hands on hips
        if all(k in landmarks for k in ['left_wrist', 'right_wrist', 'left_hip', 'right_hip']):
            left_wrist = landmarks['left_wrist']
            right_wrist = landmarks['right_wrist']
            left_hip = landmarks['left_hip']
            right_hip = landmarks['right_hip']
            
            # Hands on hips detection (wrists near hip level)
            left_hand_on_hip = abs(left_wrist['y'] - left_hip['y']) < 50
            right_hand_on_hip = abs(right_wrist['y'] - right_hip['y']) < 50
            
            analysis['hands_on_hips'] = left_hand_on_hip and right_hand_on_hip
    
    def _calculate_angle_horizontal(self, point1, point2):
        """
        Calculate angle of line between two points relative to horizontal
        """
        dx = point2['x'] - point1['x']
        dy = point2['y'] - point1['y']
        angle = math.degrees(math.atan2(dy, dx))
        return angle
    
    def _calculate_posture_score(self, analysis):
        """
        Calculate overall posture score (0-100)
        """
        score = 100
        
        # Deduct points for poor posture indicators
        if abs(analysis.get('shoulder_tilt', 0)) > 5:
            score -= 20
        
        if abs(analysis.get('head_offset_x', 0)) > 30:
            score -= 15
        
        if analysis.get('head_forward', False):
            score -= 25
        
        if abs(analysis.get('hip_tilt', 0)) > 5:
            score -= 15
        
        return max(0, score)
    
    def _classify_posture(self, analysis):
        """
        Classify the overall posture type
        """
        # Check for specific posture patterns
        if analysis.get('arms_crossed', False):
            return 'arms_crossed'
        
        if analysis.get('hands_on_hips', False):
            return 'hands_on_hips'
        
        # Check head position
        head_offset_x = analysis.get('head_offset_x', 0)
        if abs(head_offset_x) > 50:
            return 'leaning_left' if head_offset_x < 0 else 'leaning_right'
        
        if analysis.get('head_forward', False):
            return 'head_down'
        
        # Check shoulder tilt
        shoulder_tilt = abs(analysis.get('shoulder_tilt', 0))
        if shoulder_tilt > 10:
            return 'slouched'
        
        # Check posture score
        posture_score = analysis.get('posture_score', 100)
        if posture_score > 80:
            return 'upright'
        elif posture_score > 60:
            return 'relaxed'
        else:
            return 'slouched'
    
    def _calculate_pose_confidence(self, pose_landmarks):
        """
        Calculate overall confidence of pose detection
        """
        if not pose_landmarks:
            return 0.0
        
        visibilities = [landmark.visibility for landmark in pose_landmarks.landmark]
        return np.mean(visibilities)
    
    def draw_annotations(self, frame, posture_result):
        """
        Draw posture analysis annotations on the frame
        """
        if posture_result is None or not posture_result.get('body_detected', False):
            return frame
        
        annotated_frame = frame.copy()
        
        # Draw pose landmarks
        if 'pose_landmarks' in posture_result:
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                posture_result['pose_landmarks'],
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            )
        
        # Draw posture information
        posture_type = posture_result['posture_type']
        confidence = posture_result['confidence']
        posture_score = posture_result['pose_analysis'].get('posture_score', 0)
        
        # Prepare info text
        info_lines = [
            f"Posture: {self.posture_types.get(posture_type, posture_type)}",
            f"Confidence: {confidence:.2f}",
            f"Score: {posture_score:.0f}/100"
        ]
        
        # Draw info box
        y_offset = 30
        for line in info_lines:
            cv2.putText(annotated_frame, line, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(annotated_frame, line, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            y_offset += 25
        
        # Draw posture quality indicator
        self._draw_posture_indicator(annotated_frame, posture_score)
        
        return annotated_frame
    
    def _draw_posture_indicator(self, frame, posture_score):
        """
        Draw a visual indicator of posture quality
        """
        height, width = frame.shape[:2]
        
        # Position for indicator
        indicator_x = width - 50
        indicator_y = 50
        indicator_height = 200
        indicator_width = 20
        
        # Draw background bar
        cv2.rectangle(frame, 
                     (indicator_x, indicator_y), 
                     (indicator_x + indicator_width, indicator_y + indicator_height), 
                     (100, 100, 100), -1)
        
        # Draw score bar
        score_height = int((posture_score / 100) * indicator_height)
        score_y = indicator_y + indicator_height - score_height
        
        # Color based on score
        if posture_score > 80:
            color = (0, 255, 0)  # Green
        elif posture_score > 60:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 0, 255)  # Red
        
        cv2.rectangle(frame, 
                     (indicator_x, score_y), 
                     (indicator_x + indicator_width, indicator_y + indicator_height), 
                     color, -1)
        
        # Add score text
        cv2.putText(frame, f"{posture_score:.0f}", 
                   (indicator_x - 20, indicator_y + indicator_height + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def get_posture_statistics(self, posture_history):
        """
        Calculate posture statistics from history
        """
        if not posture_history:
            return {}
        
        posture_counts = {}
        total_confidence = 0
        total_score = 0
        
        for posture_data in posture_history:
            if posture_data and 'posture_type' in posture_data:
                posture_type = posture_data['posture_type']
                posture_counts[posture_type] = posture_counts.get(posture_type, 0) + 1
                total_confidence += posture_data.get('confidence', 0)
                total_score += posture_data.get('pose_analysis', {}).get('posture_score', 0)
        
        total_detections = len(posture_history)
        posture_percentages = {
            posture: (count / total_detections) * 100 
            for posture, count in posture_counts.items()
        }
        
        return {
            'posture_counts': posture_counts,
            'posture_percentages': posture_percentages,
            'average_confidence': total_confidence / total_detections if total_detections > 0 else 0,
            'average_score': total_score / total_detections if total_detections > 0 else 0,
            'total_detections': total_detections,
            'most_frequent_posture': max(posture_counts.items(), key=lambda x: x[1])[0] if posture_counts else None
        }
