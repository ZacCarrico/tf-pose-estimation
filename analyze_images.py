import logging
import os
import matplotlib.pyplot as plt
import argparse
import cv2

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator-AnalyzeImages')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fh = logging.FileHandler('info.log', "w")
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)


NECK_ANGLE_THRESHOLD = 138


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def analyze_and_save_image(args):
    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=str2bool(args.tensorrt))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368), trt_bool=str2bool(args.tensorrt))
    logger.debug('image read+')
    image = cv2.imread(args.path)
    logger.info('image=%dx%d' % (image.shape[1], image.shape[0]))

    successes = 0
    count = 0
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
        count += 1
        if angle >= NECK_ANGLE_THRESHOLD:
            successes += 1
        logger.info("good/total neck angle: " + str(round(successes / count, 2)))

    logger.debug('show+')
    # annotate image
    cv2.putText(image,
                "angle: " + str(angle),
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 2)

    plt.imshow(image)
    prefix, suffix = args.path.split('.')
    analyzed_image_path = f"{prefix}_angle{angle}.jpg"
    plt.savefig(analyzed_image_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation image analyzer')
    parser.add_argument('--path', type=str, help='if dir is provided, all images in dir are analyzed and saved to same dir')
    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help = 'if provided, resize heatmaps before they are post-processed. default=1.0')
    parser.add_argument('--model', type=str, default='mobilenet_thin',
                        help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--tensorrt', type=str, default="False",
                        help='for tensorrt process.')
    args = parser.parse_args()

    if os.path.isdir(args.path):
        files = [os.path.join(args.path, f) for f in os.listdir(args.path) if os.path.isfile(os.path.join(args.path, f))]
        print(files)
        for file in files:
            args.path = file
            analyze_and_save_image(args)
    else:
        analyze_and_save_image(args)