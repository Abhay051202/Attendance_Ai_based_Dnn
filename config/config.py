"""
Configuration file for Face Recognition Attendance System
"""

# --- MYSQL CONFIGURATION ---

# MYSQL_CONFIG = {
#     'user': 'root',           # Username
#     'password': 'Root@123',   # <--- MySQL password here
#     'host': 'localhost',      # Hostname
#     'port': 3306,             # Port
#     'database': 'demo', # The name of the DB we will create
#     'raise_on_warnings': True
# }

MYSQL_CONFIG = {
    'user': 'team',
    'password': 'Team@123',
    'host': '192.168.1.106',  
    'port': 3306,
    'database': 'demo',
}

# --- SMTP EMAIL CONFIGURATION ---

SMTP_CONFIG = {
    'smtp_server': 'smtp.gmail.com',      # SMTP server (Gmail example)
    'smtp_port': 587,                      # Port (587 for TLS, 465 for SSL)
    'sender_email': 'your_email@gmail.com',  # Your email address
    'sender_password': 'your_app_password',  # App-specific password
    'use_tls': True,                       # Use TLS encryption
    'admin_email': 'admin@example.com',    # Admin email for notifications
}

# Email Notification Settings
EMAIL_NOTIFICATIONS_ENABLED = False        # Enable/disable email notifications
SEND_ARRIVAL_NOTIFICATIONS = False        # Send email on every arrival (not recommended)
SEND_DEPARTURE_NOTIFICATIONS = False      # Send email on every departure (not recommended)
SEND_ABSENCE_ALERTS = True               # Send alerts for absent employees
SEND_LATE_ARRIVAL_ALERTS = True          # Send alerts for late arrivals
SEND_DAILY_SUMMARY = False               # Send daily summary emails (all attendance data)
DAILY_SUMMARY_TIME = '18:00'             # Time to send daily summary (24-hour format)


# --- FILE PATHS ---

FACE_ENCODINGS_PATH = 'data/face_encodings.pkl' 
UNKNOWN_FACES_DIR = 'data/unknown_faces' 

# Face Recognition Settings
SIMILARITY_THRESHOLD = 0.5  
DETECTION_SIZE = (1024, 1024) 
FACE_DETECTION_MODEL = 'buffalo_l' 

# Execution Providers (GPU/CPU)
EXECUTION_PROVIDERS = [
    'CUDAExecutionProvider',
    'CPUExecutionProvider'
]

# ByteTrack Configuration
TRACK_ACTIVATION_THRESHOLD = 0.5  
LOST_TRACK_BUFFER = 30            
FRAME_RATE = 30                   

# Attendance Settings
ATTENDANCE_COOLDOWN_SECONDS = 20   # Set to 20 or higher for logic to work well
AUTO_MARK_ARRIVAL = True          
AUTO_MARK_LEAVING = True          

# Video Processing Settings
# Video Processing Settings
# WEBCAM_INDEX = 0   
WEBCAM_INDEX = "rtsp://admin:Prosper%404321@192.168.1.240:554/cam/realmonitor?channel=8&subtype=0"               
DISPLAY_LANDMARKS = True          
DISPLAY_FPS = True                
DISPLAY_INFO_PANEL = True

# Performance Optimization
PROCESS_EVERY_N_FRAMES = 5    # Run Face AI every Nth frame (Increase if laggy)
RESIZE_FACTOR = 0.5           # Resize frame for AI (0.5 = 50% size, faster)         

# Annotation Settings
BOX_THICKNESS = 2
TEXT_THICKNESS = 1
TEXT_SCALE = 0.5
TEXT_PADDING = 5

# Colors (BGR format)
COLOR_DETECTED = (0, 255, 0)      
COLOR_UNKNOWN = (0, 0, 255)       
COLOR_WARNING = (0, 165, 255)     
COLOR_INFO = (255, 255, 255)      

# CSV Export Settings
CSV_EXPORT_FOLDER = 'exports/'
CSV_DATE_FORMAT = '%Y-%m-%d'
CSV_TIME_FORMAT = '%H:%M:%S'

# Registration Settings
REGISTRATION_CAPTURE_KEY = 'c'   
REGISTRATION_CANCEL_KEY = 'q'    
MIN_FACE_SIZE = 50                
MAX_REGISTRATION_ATTEMPTS = 3     

# Logging Settings
LOG_ATTENDANCE_MARKS = True       
LOG_RECOGNITION_EVENTS = True     
LOG_ERRORS = True                 

def get_config():
    """Return configuration as dictionary"""
    return {
        # UPDATED: Returns MySQL config instead of file path
        'mysql_config': MYSQL_CONFIG,
        'smtp_config': SMTP_CONFIG,
        'face_encodings_path': FACE_ENCODINGS_PATH,
        'similarity_threshold': SIMILARITY_THRESHOLD,
        'detection_size': DETECTION_SIZE,
        'face_detection_model': FACE_DETECTION_MODEL,
        'execution_providers': EXECUTION_PROVIDERS,
        'track_activation_threshold': TRACK_ACTIVATION_THRESHOLD,
        'lost_track_buffer': LOST_TRACK_BUFFER,
        'frame_rate': FRAME_RATE,
        'attendance_cooldown': ATTENDANCE_COOLDOWN_SECONDS,
        'start_cooldown': ATTENDANCE_COOLDOWN_SECONDS, # Alias for clarity
        'webcam_index': WEBCAM_INDEX,
        'display_landmarks': DISPLAY_LANDMARKS,
        'display_fps': DISPLAY_FPS,
        'process_every_n_frames': PROCESS_EVERY_N_FRAMES,
        'resize_factor': RESIZE_FACTOR,
    }

def validate_config():
    """Validate configuration settings"""
    errors = []
    
    if not 0.0 <= SIMILARITY_THRESHOLD <= 1.0:
        errors.append("SIMILARITY_THRESHOLD must be between 0.0 and 1.0")
    
    if ATTENDANCE_COOLDOWN_SECONDS < 0:
        errors.append("ATTENDANCE_COOLDOWN_SECONDS must be non-negative")
        
    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(errors))
    
    return True

try:
    validate_config()
except ValueError as e:
    print(f"Warning: {e}")
    print("Using default values...")