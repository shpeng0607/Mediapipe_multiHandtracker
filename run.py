import cv2
import numpy as np
import time
import os
import requests

from PIL import Image
from src.hand_tracker import HandTracker
from process_keypoints import *
from drawing_tool import *

PALM_MODEL_PATH = "./models/palm_detection_builtin.pb"
LANDMARK_MODEL_PATH = "./models/hand_landmark_small.pb"
ANCHORS_PATH = "./models/anchors.csv"
FILTER_COEFFS = [-0.17857, -0.07143, 0.03571, 0.14286, 0.25, 0.35714, 0.46429]

cv2.namedWindow("Hand Tracking")
capture = cv2.VideoCapture(0)
if capture.isOpened():
    hasFrame, frame = capture.read()
else:
    hasFrame = False

detector = HandTracker(
    PALM_MODEL_PATH,
    LANDMARK_MODEL_PATH,
    ANCHORS_PATH,
)

def process_hand(frame, hand_data, hand_name, hand_history):
    box = hand_data["box"]
    palm_points = hand_data["palm"]
    angle = hand_data["angle"]
    points = hand_data["joints"][:,:2]

    if box is not None:
        draw_box(frame, box)

    if palm_points is not None:
        pass
        # draw_keypoints(frame, palm_points)

    if points is not None:
        if hand_name == "left":
            box_centre = hand_data["joints"][5,:2]#detector.palm_keypoints.astype(float).mean(0)
            #box_centre = np.array(box.mean(axis=0)).astype(float)

            angle = angle - 10 #orientation - 90 - 10
            x_shift = 0#+np.sin(np.deg2rad(angle)) * 30 * -2
            y_shift = 0#+np.cos(np.deg2rad(angle)) * 30 * -2
            pointing = box_centre + np.array([x_shift, y_shift])
            hand_history.append(pointing)
        else:
            pass

        draw_keypoints(frame, points)
        draw_connections(frame, points)

left_history = []
right_history = []
while hasFrame:
    frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detector(image)

    if detector.left_hand["joints"] is not None:
        process_hand(frame, detector.left_hand, "left", left_history)
    else:
        left_history = []

    if detector.right_hand["joints"] is not None:
        process_hand(frame, detector.right_hand, "right", right_history)
    else:
        right_history = []

    cv2.imshow("Hand Tracking", frame)
    hasFrame, frame = capture.read()

    key = cv2.waitKey(1)
    if key == 27:
        break

capture.release()
cv2.destroyAllWindows()