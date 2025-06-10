import cv2
import numpy as np
from deepface import DeepFace
import logging

class EmotionDetector:
    def __init__(self):
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def detect_emotions(self, frame, sensitivity=0.7):
        """
        Detect emotions in the given frame using DeepFace
        
        Args:
            frame: Input image frame
            sensitivity: Confidence threshold for emotion detection
            
        Returns:
            Dictionary containing emotion analysis results
        """
        try:
            # Convert BGR to RGB for DeepFace
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces first
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return None
            
            # Use the largest face detected
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            
            # Extract face region
            face_roi = rgb_frame[y:y+h, x:x+w]
            
            if face_roi.size == 0:
                return None
            
            # Analyze emotions using DeepFace
            result = DeepFace.analyze(
                face_roi, 
                actions=['emotion'], 
                enforce_detection=False,
                silent=True
            )
            
            if isinstance(result, list):
                result = result[0]
            
            emotions = result['emotion']
            dominant_emotion = result['dominant_emotion']
            
            # Filter emotions by sensitivity threshold
            filtered_emotions = {k: v for k, v in emotions.items() 
                               if v/100.0 >= sensitivity}
            
            return {
                'dominant_emotion': dominant_emotion,
                'confidence': emotions[dominant_emotion] / 100.0,
                'all_emotions': {k: v/100.0 for k, v in emotions.items()},
                'filtered_emotions': {k: v/100.0 for k, v in filtered_emotions.items()},
                'face_coordinates': (x, y, w, h),
                'face_detected': True
            }
            
        except Exception as e:
            logging.error(f"Error in emotion detection: {str(e)}")
            return None
    
    def draw_annotations(self, frame, emotion_result):
        """
        Draw emotion detection annotations on the frame
        
        Args:
            frame: Input image frame
            emotion_result: Result from detect_emotions method
            
        Returns:
            Annotated frame
        """
        if emotion_result is None or not emotion_result.get('face_detected', False):
            return frame
        
        annotated_frame = frame.copy()
        x, y, w, h = emotion_result['face_coordinates']
        
        # Draw face rectangle
        cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Prepare emotion text
        dominant_emotion = emotion_result['dominant_emotion']
        confidence = emotion_result['confidence']
        
        # Draw emotion label
        label = f"{dominant_emotion}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        
        # Draw background rectangle for text
        cv2.rectangle(annotated_frame, 
                     (x, y - label_size[1] - 10), 
                     (x + label_size[0], y), 
                     (0, 255, 0), -1)
        
        # Draw text
        cv2.putText(annotated_frame, label, (x, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Draw emotion bar chart on the side
        self._draw_emotion_chart(annotated_frame, emotion_result['all_emotions'])
        
        return annotated_frame
    
    def _draw_emotion_chart(self, frame, emotions):
        """
        Draw a small emotion intensity chart on the frame
        """
        height, width = frame.shape[:2]
        chart_x = width - 200
        chart_y = 50
        bar_width = 20
        bar_spacing = 25
        
        # Sort emotions by intensity
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        
        for i, (emotion, intensity) in enumerate(sorted_emotions[:5]):
            y_pos = chart_y + i * bar_spacing
            bar_length = int(intensity * 150)
            
            # Draw bar
            color = self._get_emotion_color(emotion)
            cv2.rectangle(frame, 
                         (chart_x, y_pos), 
                         (chart_x + bar_length, y_pos + bar_width), 
                         color, -1)
            
            # Draw emotion name
            cv2.putText(frame, f"{emotion[:4]}", 
                       (chart_x - 50, y_pos + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def _get_emotion_color(self, emotion):
        """
        Get BGR color for emotion visualization
        """
        colors = {
            'happy': (0, 255, 255),     # Yellow
            'sad': (255, 0, 0),         # Blue
            'angry': (0, 0, 255),       # Red
            'fear': (128, 0, 128),      # Purple
            'surprise': (255, 192, 203), # Pink
            'disgust': (0, 255, 0),     # Green
            'neutral': (128, 128, 128)  # Gray
        }
        return colors.get(emotion.lower(), (128, 128, 128))
    
    def get_emotion_statistics(self, emotion_history):
        """
        Calculate emotion statistics from history
        
        Args:
            emotion_history: List of emotion results
            
        Returns:
            Dictionary with emotion statistics
        """
        if not emotion_history:
            return {}
        
        # Count emotion occurrences
        emotion_counts = {}
        total_confidence = 0
        
        for emotion_data in emotion_history:
            if emotion_data and 'dominant_emotion' in emotion_data:
                emotion = emotion_data['dominant_emotion']
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                total_confidence += emotion_data.get('confidence', 0)
        
        # Calculate percentages
        total_detections = len(emotion_history)
        emotion_percentages = {
            emotion: (count / total_detections) * 100 
            for emotion, count in emotion_counts.items()
        }
        
        return {
            'emotion_counts': emotion_counts,
            'emotion_percentages': emotion_percentages,
            'average_confidence': total_confidence / total_detections if total_detections > 0 else 0,
            'total_detections': total_detections,
            'most_frequent_emotion': max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else None
        }
