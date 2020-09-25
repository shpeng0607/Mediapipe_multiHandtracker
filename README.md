# Mediapipe_multiHandtracker
MultiHandtracker using Mediapipe models in python <br>
Self-practice

# Getting Started
Requirements
------------
* These are required package to use this project
```
numpy
opencv-python
tensorflow
matplotlib
```

* run
```
$ python run.py
```
# Structure
* models
  * anchors.csv
  * hand_landmark_small.pb
  * palm_detection_builtin.pb
* src
  * hand_tracker.py
  * non_maximum_suppression.py
* drawing_tool.py
* process_keypoints.py
* run.py

# References
1. [mediapipe](https://github.com/google/mediapipe) by google
2. [hand_tracking](https://github.com/metalwhale/hand_tracking) by metalwhale
3. [Multi-HandTrackingGPU](https://github.com/TviNet/Multi-HandTrackingGPU) by TviNet
