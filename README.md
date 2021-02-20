Hey! This is a clone of tf-pose-estimation with TensorFlow2 from gseth2409, who cloned it from 
tf-pose-estimation by Ildoo Kim, who wrote it in TensorFlow1. 
Link to original repo (TensorFlow1): https://www.github.com/ildoonet/tf-openpose
Link to the Tensorflow2 repo: https://github.com/gsethi2409/tf-pose-estimation

This repo is used to measure pose. See https://www.github.com/ildoonet/tf-openpose

I modified this to calculate neck angle for the purpose of improving posture while working on a computer.
It works surprisingly well for letting you know when your posture has deteriorated.
If you don't sit up straight, the image window will pop up. Once your posture improves, it will
stop popping up. There is a bit of delay between when the image is acquired and when it pops up (< 1 s).

To run this with a webcam, position the webcam so it's facing you from the side and run
`python run_webcam.py --model=mobilenet_thin --resize=432x368 --camera=0`
You may have to adjust the `--camera` integer.

