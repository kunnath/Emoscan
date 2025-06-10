"""
Configuration settings for EmoScan application
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
EXPORTS_DIR = BASE_DIR / "exports"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
EXPORTS_DIR.mkdir(exist_ok=True)

# Camera settings
CAMERA_CONFIG = {
    'default_width': 640,
    'default_height': 480,
    'default_fps': 30,
    'ip_camera_fps': 15,
    'buffer_size': 1,
    'connection_timeout': 10,
    'max_retries': 3
}

# Emotion detection settings
EMOTION_CONFIG = {
    'models': ['VGG-Face', 'Facenet', 'OpenFace', 'DeepFace'],
    'default_model': 'VGG-Face',
    'detector_backend': 'opencv',
    'enforce_detection': False,
    'align': True,
    'normalization': 'base',
    'silent': True
}

# Pose detection settings
POSE_CONFIG = {
    'model_complexity': 1,
    'enable_segmentation': False,
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5,
    'static_image_mode': False
}

# Performance settings
PERFORMANCE_CONFIG = {
    'process_every_nth_frame': 3,
    'chart_update_frequency': 15,
    'max_history_length': 100,
    'enable_gpu': True,
    'memory_fraction': 0.7
}

# UI settings
UI_CONFIG = {
    'page_title': "EmoScan - Emotion & Body Language Analyzer",
    'page_icon': "🎭",
    'layout': "wide",
    'sidebar_state': "expanded",
    'theme': {
        'primary_color': "#4CAF50",
        'background_color': "#FFFFFF",
        'secondary_background_color': "#F0F2F6",
        'text_color': "#262730"
    }
}

# Visualization settings
VIZ_CONFIG = {
    'emotion_colors': {
        'happy': '#FFD700',
        'sad': '#4169E1',
        'angry': '#FF4500',
        'fear': '#8B008B',
        'surprise': '#FF69B4',
        'disgust': '#32CD32',
        'neutral': '#808080'
    },
    'posture_colors': {
        'upright': '#00FF00',
        'slouched': '#FF4500',
        'leaning_left': '#FFA500',
        'leaning_right': '#FFA500',
        'head_down': '#FF6B6B',
        'arms_crossed': '#9370DB',
        'hands_on_hips': '#20B2AA',
        'relaxed': '#98FB98'
    },
    'chart_height': 400,
    'update_interval': 1.0
}

# Logging settings
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_handler': True,
    'console_handler': True,
    'log_file': DATA_DIR / 'emoscan.log'
}

# Data export settings
EXPORT_CONFIG = {
    'default_filename': 'emoscan_data.csv',
    'include_raw_scores': True,
    'include_timestamps': True,
    'datetime_format': '%Y-%m-%d %H:%M:%S',
    'csv_separator': ','
}

# Security and privacy settings
SECURITY_CONFIG = {
    'save_frames': False,
    'encrypt_data': False,
    'max_session_duration': 3600,  # seconds
    'auto_cleanup': True
}

# Default analysis thresholds
ANALYSIS_THRESHOLDS = {
    'emotion_confidence': 0.7,
    'pose_confidence': 0.6,
    'good_posture_score': 80,
    'fair_posture_score': 60,
    'poor_posture_score': 40
}

# Phone camera app configurations
PHONE_CAMERA_CONFIGS = {
    'ip_webcam_android': {
        'default_port': 8080,
        'video_path': '/video',
        'mjpeg_path': '/videofeed',
        'settings_path': '/settings'
    },
    'epoccam_ios': {
        'default_port': 80,
        'video_path': '/live',
        'mjpeg_path': '/mjpeg'
    },
    'droidcam': {
        'default_port': 4747,
        'video_path': '/mjpegfeed'
    }
}

# WiFi camera configurations
WIFI_CAMERA_CONFIGS = {
    'generic_mjpeg': {
        'url_format': 'http://{ip}:{port}/video',
        'default_port': 8080
    },
    'generic_rtsp': {
        'url_format': 'rtsp://{ip}:{port}/stream',
        'default_port': 554
    },
    'hikvision': {
        'url_format': 'rtsp://{username}:{password}@{ip}:{port}/Streaming/Channels/1',
        'default_port': 554
    },
    'dahua': {
        'url_format': 'rtsp://{username}:{password}@{ip}:{port}/cam/realmonitor?channel=1&subtype=0',
        'default_port': 554
    }
}

# Environment variables
def get_env_config():
    """Get configuration from environment variables"""
    return {
        'debug_mode': os.getenv('EMOSCAN_DEBUG', 'False').lower() == 'true',
        'gpu_enabled': os.getenv('EMOSCAN_GPU', 'True').lower() == 'true',
        'model_cache_dir': os.getenv('EMOSCAN_MODEL_CACHE', str(MODELS_DIR)),
        'data_dir': os.getenv('EMOSCAN_DATA_DIR', str(DATA_DIR)),
        'log_level': os.getenv('EMOSCAN_LOG_LEVEL', 'INFO'),
        'max_concurrent_users': int(os.getenv('EMOSCAN_MAX_USERS', '5')),
        'enable_telemetry': os.getenv('EMOSCAN_TELEMETRY', 'False').lower() == 'true'
    }

# Model download URLs (for manual installation if needed)
MODEL_URLS = {
    'deepface_models': [
        'https://github.com/serengil/deepface_models/releases/download/v1.0/vgg_face_weights.h5',
        'https://github.com/serengil/deepface_models/releases/download/v1.0/facenet_weights.h5'
    ],
    'mediapipe_models': [
        'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task'
    ]
}

# System requirements check
SYSTEM_REQUIREMENTS = {
    'min_python_version': (3, 8),
    'min_ram_gb': 4,
    'min_disk_space_gb': 2,
    'recommended_ram_gb': 8,
    'recommended_disk_space_gb': 5,
    'supported_platforms': ['Windows', 'macOS', 'Linux'],
    'required_packages': [
        'streamlit>=1.28.0',
        'opencv-python>=4.8.0',
        'numpy>=1.24.0',
        'pandas>=2.0.0'
    ],
    'optional_packages': [
        'deepface>=0.0.79',
        'mediapipe>=0.10.7',
        'plotly>=5.17.0',
        'tensorflow>=2.13.0'
    ]
}
