import logging
import queue as Queue
import sqlite3
import threading
import time
import argparse
import os

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

# Console handler (optional)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

logger.info(f'Logging to: {log_file}')

NECK_ANGLE_THRESHOLD = 138

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def detect_cameras():
    """Detect available cameras"""
    available_cameras = []
    for i in range(10):  # Check first 10 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                available_cameras.append(i)
            cap.release()
    return available_cameras

# bufferless VideoCapture
class VideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = Queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()
                except Queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()

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
        self.selected_camera = 0  # Default camera
        self.available_cameras = []
        
        # Detect available cameras
        self.detect_cameras_on_startup()
        
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
    
    def detect_cameras_on_startup(self):
        """Detect available cameras on startup"""
        logger.info("Detecting available cameras...")
        self.available_cameras = detect_cameras()
        if not self.available_cameras:
            self.available_cameras = [0]  # Fallback to camera 0
        logger.info(f"Available cameras: {self.available_cameras}")
    
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
        # Extract camera ID from menu title
        cam_id = int(sender.title.split()[-1])
        
        # Update selected camera
        old_camera = self.selected_camera
        self.selected_camera = cam_id
        
        # Update menu checkmarks
        for item in self.camera_menu.values():
            item.state = False
        sender.state = True
        
        logger.info(f"Camera changed from {old_camera} to {cam_id}")
        
        # If monitoring is running, restart with new camera
        if self.running:
            logger.info("Restarting monitoring with new camera...")
            self.stop_monitoring(None)
            time.sleep(1)
            self.start_monitoring(None)
        
    def setup_database(self):
        """Initialize SQLite database"""
        self.sql_conn = sqlite3.connect('posture.db', check_same_thread=False)
        self.sql_cursor = self.sql_conn.cursor()
        self.sql_cursor.execute('''
            CREATE TABLE IF NOT EXISTS neck_angle (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                angle INTEGER NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.sql_conn.commit()
    
    def setup_camera_and_estimator(self):
        """Initialize camera and pose estimator"""
        try:
            # Setup pose estimator
            model = 'mobilenet_thin'
            resize = '432x368'
            w, h = model_wh(resize)
            
            if w > 0 and h > 0:
                self.estimator = TfPoseEstimator(get_graph_path(model), target_size=(w, h), trt_bool=False)
            else:
                self.estimator = TfPoseEstimator(get_graph_path(model), target_size=(432, 368), trt_bool=False)
            
            # Setup camera
            self.camera = VideoCapture(self.selected_camera)
            test_image = self.camera.read()
            logger.info(f'Camera initialized: {test_image.shape[1]}x{test_image.shape[0]}')
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup camera/estimator: {e}")
            return False
    
    def monitoring_loop(self):
        """Main monitoring loop that runs in a separate thread"""
        while self.running:
            try:
                image = self.camera.read()
                
                # Pose estimation
                humans = self.estimator.inference(image, resize_to_default=True, upsample_size=4.0)
                image, angle = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
                
                if angle:
                    self.current_angle = int(angle[0])
                else:
                    self.current_angle = -1
                    
                # Update statistics
                if self.current_angle != -1:
                    self.sql_cursor.execute(
                        "INSERT INTO neck_angle(angle) VALUES (?)", (self.current_angle,)
                    )
                    self.sql_conn.commit()
                    self.total_count += 1
                    
                    if self.current_angle >= NECK_ANGLE_THRESHOLD:
                        self.good_posture_count += 1
                    
                    # Update menu items
                    good_percentage = round((self.good_posture_count / self.total_count) * 100, 1)
                    self.angle_item.title = f"Current Angle: {self.current_angle}°"
                    self.stats_item.title = f"Good Posture: {good_percentage}%"
                    
                    # Show warning window for poor posture
                    if self.current_angle < NECK_ANGLE_THRESHOLD:
                        self.title = "🔴"  # Red circle for poor posture
                        # Show webcam window like in run_webcam.py
                        cv2.imshow('tf-pose-estimation result', image)
                        cv2.waitKey(1)  # Process the window event
                        time.sleep(2)
                        cv2.destroyAllWindows()
                        cv2.waitKey(1)  # Ensure the destroy event is processed
                    else:
                        self.title = "🟢"  # Green circle for good posture
                        
                else:
                    self.angle_item.title = "Current Angle: No person detected"
                    self.title = "⚪"  # White circle when no person detected
                
                time.sleep(0.1)  # Small delay to prevent excessive CPU usage
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1)
    
    @rumps.clicked("Start Monitoring")
    def start_monitoring(self, _):
        if not self.running:
            logger.info("Starting posture monitoring...")
            
            # Setup components
            if not self.setup_camera_and_estimator():
                rumps.alert("Error", "Failed to initialize camera or pose estimator")
                return
                
            self.setup_database()
            
            # Start monitoring
            self.running = True
            self.monitoring_thread = threading.Thread(target=self.monitoring_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            
            # Update UI
            self.status_item.title = "Status: Running"
            self.start_button.title = "Start Monitoring (Running)"
            self.title = "⚪"  # White circle when starting
            
            logger.info("Posture monitoring started")
    
    @rumps.clicked("Stop Monitoring") 
    def stop_monitoring(self, _):
        if self.running:
            logger.info("Stopping posture monitoring...")
            self.running = False
            
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=3)
            
            # Clean up
            if self.sql_conn:
                try:
                    self.sql_conn.close()
                except:
                    pass
                    
            cv2.destroyAllWindows()
            
            # Update UI
            self.status_item.title = "Status: Not Running"
            self.angle_item.title = "Current Angle: --"
            self.stats_item.title = "Good Posture: --%"
            self.start_button.title = "Start Monitoring"
            self.title = "⚪"  # White circle when stopped
            
            logger.info("Posture monitoring stopped")

if __name__ == "__main__":
    app = PostureApp()
    app.run()