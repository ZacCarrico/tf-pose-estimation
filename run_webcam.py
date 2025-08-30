import logging
import queue as Queue
import sqlite3
import threading
import time

import argparse
import cv2

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
sql_conn = sqlite3.connect('posture.db')
sql_cursor = sql_conn.cursor()

# Create the neck_angle table if it doesn't exist
sql_cursor.execute('''
    CREATE TABLE IF NOT EXISTS neck_angle (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        angle INTEGER NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
''')
sql_conn.commit()

fh = logging.FileHandler('info.log', "w")
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)

fps_time = 0

NECK_ANGLE_THRESHOLD = 138

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


# bufferless VideoCapture
class VideoCapture:
    """
    class copied from
    https://stackoverflow.com/questions/54460797/how-to-disable-buffer-in-opencv-camera
    """

    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = Queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

  # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
      while True:
        ret, frame = self.cap.read()
        if not ret:
          break
        if not self.q.empty():
          try:
            self.q.get_nowait()   # discard previous (unprocessed) frame
          except Queue.Empty:
            pass
        self.q.put(frame)

    def read(self):
      return self.q.get()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    
    parser.add_argument('--tensorrt', type=str, default="False",
                        help='for tensorrt process.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=str2bool(args.tensorrt))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368), trt_bool=str2bool(args.tensorrt))
    logger.debug('cam read+')
    cam = VideoCapture(args.camera)
    image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

    successes = 0
    count = 0
    while True:
        image = cam.read()
        logger.debug('image process+')

        # this takes about 0.3s on a CPU
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

        logger.debug('postprocess+')
        image, angle = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        if angle:
            angle = int(angle[0])
        else:
            angle = -1
        if angle != -1:
            sql_cursor.execute(
                """INSERT INTO neck_angle(angle) VALUES ({})""".format(angle)
            )
            sql_conn.commit()
            count += 1
            if angle >= NECK_ANGLE_THRESHOLD:
                successes += 1
            logger.info("good/total neck angle: " + str(round(successes / count, 2)))

        logger.debug('show+')
        cv2.putText(image,
                    "angle: " + str(angle) + ", FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        if angle != -1 and angle < NECK_ANGLE_THRESHOLD:
            cv2.imshow('tf-pose-estimation result', image)
            if cv2.waitKey(1) == 27:
                break
            time.sleep(2)
            cv2.destroyAllWindows()

    sql_conn.close()
    cv2.destroyAllWindows()
