import logging
import queue as Queue
import sqlite3
import threading
import time
import os
import json
import signal
import sys

import cv2
import rumps

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

# Setup logging to standard macOS location
log_dir = os.path.expanduser('~/Library/Logs/PostureMonitor')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'posture_monitor.log')

logger = logging.getLogger('PostureMenuBar')
logger.setLevel(logging.INFO)

# File handler
fh = logging.FileHandler(log_file)
fh.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

# Console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

logger.info(f'Logging to: {log_file}')

NECK_ANGLE_THRESHOLD = 138

# Settings file location
SETTINGS_DIR = os.path.expanduser('~/Library/Application Support/PostureMonitor')
SETTINGS_FILE = os.path.join(SETTINGS_DIR, 'settings.json')
os.makedirs(SETTINGS_DIR, exist_ok=True)

def load_settings():
    """Load settings from file"""
    default_settings = {'default_camera': 0}
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as f:
                return {**default_settings, **json.load(f)}
    except Exception as e:
        logger.warning(f"Failed to load settings: {e}")
    return default_settings

def save_settings(settings):
    """Save settings to file"""
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save settings: {e}")

def detect_cameras():
    """Detect available cameras"""
    available_cameras = []
    for i in range(3):  # Check first 3 camera indices only
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    available_cameras.append(i)
                    logger.info(f"Found camera {i}")
            cap.release()
        except Exception as e:
            logger.warning(f"Error checking camera {i}: {e}")
    return available_cameras if available_cameras else [0]

# Simple VideoCapture without threading
class SimpleVideoCapture:
    def __init__(self, camera_id):
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
    def read(self):
        ret, frame = self.cap.read()
        return frame if ret else None
        
    def release(self):
        if self.cap:
            self.cap.release()

class PostureApp(rumps.App):
    def __init__(self):
        super(PostureApp, self).__init__("⚪", title="Posture Monitor")
        self.running = False
        self.monitoring_thread = None
        self.camera = None
        self.estimator = None
        self.sql_conn = None
        self.sql_cursor = None
        self.current_angle = -1
        self.good_posture_count = 0
        self.total_count = 0
        self.settings = load_settings()
        self.selected_camera = self.settings.get('default_camera', 0)
        self.available_cameras = detect_cameras()
        
        # Menu items
        self.start_button = rumps.MenuItem("Start Monitoring", callback=self.start_monitoring)
        self.stop_button = rumps.MenuItem("Stop Monitoring", callback=self.stop_monitoring)
        self.status_item = rumps.MenuItem("Status: Not Running", callback=None)
        self.angle_item = rumps.MenuItem("Current Angle: --", callback=None)
        self.stats_item = rumps.MenuItem("Good Posture: --%", callback=None)
        
        # Camera selection submenu
        self.camera_menu = rumps.MenuItem("Select Camera")
        self.setup_camera_menu()
        
        # Add menu items
        self.menu = [
            self.start_button,
            self.stop_button,
            rumps.separator,
            self.camera_menu,
            rumps.separator,
            self.status_item,
            self.angle_item, 
            self.stats_item
        ]
        
        # Setup signal handler for clean exit
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle system signals for clean shutdown"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.cleanup_and_exit()
    
    def setup_camera_menu(self):
        """Setup camera selection submenu"""
        camera_items = []
        for cam_id in self.available_cameras:
            item = rumps.MenuItem(f"Camera {cam_id}", callback=self.select_camera)
            if cam_id == self.selected_camera:
                item.state = True
            camera_items.append(item)
        self.camera_menu.update(camera_items)
    
    def select_camera(self, sender):
        """Handle camera selection"""
        cam_id = int(sender.title.split()[-1])
        
        old_camera = self.selected_camera
        self.selected_camera = cam_id
        
        # Save to settings
        self.settings['default_camera'] = cam_id
        save_settings(self.settings)
        
        # Update menu checkmarks
        for item in self.camera_menu.values():
            item.state = False
        sender.state = True
        
        logger.info(f"Camera changed from {old_camera} to {cam_id}")
    
    def setup_components(self):
        """Initialize camera, estimator and database"""
        try:
            # Setup database
            self.sql_conn = sqlite3.connect('posture.db')
            self.sql_cursor = self.sql_conn.cursor()
            self.sql_cursor.execute('''
                CREATE TABLE IF NOT EXISTS neck_angle (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    angle INTEGER NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            self.sql_conn.commit()
            
            # Setup pose estimator
            model = 'mobilenet_thin'
            resize = '432x368'
            w, h = model_wh(resize)
            
            if w > 0 and h > 0:
                self.estimator = TfPoseEstimator(get_graph_path(model), target_size=(w, h), trt_bool=False)
            else:
                self.estimator = TfPoseEstimator(get_graph_path(model), target_size=(432, 368), trt_bool=False)
            
            # Setup camera
            self.camera = SimpleVideoCapture(self.selected_camera)
            test_image = self.camera.read()
            if test_image is None:
                raise Exception("Failed to read from camera")
                
            logger.info(f'Camera initialized: {test_image.shape[1]}x{test_image.shape[0]}')
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup components: {e}")
            return False
    
    def monitoring_loop(self):
        """Simplified monitoring loop"""
        logger.info("Starting monitoring loop")
        consecutive_errors = 0
        
        while self.running:
            try:
                image = self.camera.read()
                if image is None:
                    consecutive_errors += 1
                    if consecutive_errors > 5:
                        logger.error("Too many camera read failures")
                        break
                    time.sleep(0.5)
                    continue
                
                consecutive_errors = 0
                
                # Pose estimation
                humans = self.estimator.inference(image, resize_to_default=True, upsample_size=4.0)
                image, angle = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
                
                if angle:
                    self.current_angle = int(angle[0])
                else:
                    self.current_angle = -1
                
                # Add text overlay
                cv2.putText(image, f"Angle: {self.current_angle}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Update statistics and UI
                if self.current_angle != -1:
                    try:
                        self.sql_cursor.execute("INSERT INTO neck_angle(angle) VALUES (?)", (self.current_angle,))
                        self.sql_conn.commit()
                    except Exception as e:
                        logger.error(f"Database error: {e}")
                        
                    self.total_count += 1
                    if self.current_angle >= NECK_ANGLE_THRESHOLD:
                        self.good_posture_count += 1
                    
                    good_percentage = round((self.good_posture_count / self.total_count) * 100, 1)
                    self.angle_item.title = f"Current Angle: {self.current_angle}°"
                    self.stats_item.title = f"Good Posture: {good_percentage}%"
                    
                    # Show window for poor posture - SIMPLIFIED approach
                    if self.current_angle < NECK_ANGLE_THRESHOLD:
                        self.title = "🔴"
                        # Show window immediately and briefly
                        cv2.namedWindow('Posture Alert', cv2.WINDOW_NORMAL)
                        cv2.imshow('Posture Alert', image)
                        cv2.waitKey(1)
                        
                        # Brief pause, checking for stop signal
                        for _ in range(10):  # 1 second total
                            if not self.running:
                                break
                            time.sleep(0.1)
                        
                        cv2.destroyWindow('Posture Alert')
                        cv2.waitKey(1)
                    else:
                        self.title = "🟢"
                else:
                    self.angle_item.title = "Current Angle: No person detected"
                    self.title = "⚪"
                
                # Small delay
                time.sleep(0.2)
                
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Monitoring loop error: {e}")
                if consecutive_errors > 10:
                    logger.error("Too many errors, stopping")
                    break
                time.sleep(1)
        
        logger.info("Monitoring loop ended")
    
    @rumps.clicked("Start Monitoring")
    def start_monitoring(self, _):
        if not self.running:
            logger.info("Starting monitoring...")
            
            if not self.setup_components():
                rumps.alert("Error", "Failed to initialize components")
                return
            
            self.running = True
            self.monitoring_thread = threading.Thread(target=self.monitoring_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            
            self.status_item.title = "Status: Running"
            self.start_button.title = "Start Monitoring (Running)"
            self.title = "⚪"
            
            logger.info("Monitoring started")
    
    @rumps.clicked("Stop Monitoring")
    def stop_monitoring(self, _):
        if self.running:
            logger.info("Stopping monitoring...")
            self.running = False
            
            # Wait a moment for thread to notice
            time.sleep(0.3)
            
            # Clean shutdown
            self.cleanup_resources()
            
            # Wait for thread
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=1)
            
            self.status_item.title = "Status: Not Running"
            self.angle_item.title = "Current Angle: --"
            self.stats_item.title = "Good Posture: --%"
            self.start_button.title = "Start Monitoring"
            self.title = "⚪"
            
            logger.info("Monitoring stopped")
    
    def cleanup_resources(self):
        """Clean up all resources safely"""
        try:
            # Close any OpenCV windows
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except:
            pass
            
        try:
            if self.camera:
                self.camera.release()
                self.camera = None
        except Exception as e:
            logger.error(f"Camera cleanup error: {e}")
        
        try:
            if self.sql_conn:
                self.sql_conn.close()
                self.sql_conn = None
        except Exception as e:
            logger.error(f"Database cleanup error: {e}")
    
    def cleanup_and_exit(self):
        """Final cleanup and exit"""
        if self.running:
            self.running = False
            time.sleep(0.5)
        self.cleanup_resources()
        sys.exit(0)

if __name__ == "__main__":
    try:
        app = PostureApp()
        app.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"App error: {e}")
    finally:
        cv2.destroyAllWindows()