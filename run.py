import cv2
import numpy as np
import time

# from PIL import Image
from src.hand_tracker import HandTracker
from drawing_tool import *

class main:

    def __init__(self):
        PALM_MODEL_PATH = "./models/palm_detection_builtin.pb"
        LANDMARK_MODEL_PATH = "./models/hand_landmark_small.pb"
        ANCHORS_PATH = "./models/anchors.csv"
        FILTER_COEFFS = [-0.17857, -0.07143, 0.03571, 0.14286, 0.25, 0.35714, 0.46429]
        self.last_Rhandpoint = None
        self.last_Lhandpoint = None
        self.detector = HandTracker(
            PALM_MODEL_PATH,
            LANDMARK_MODEL_PATH,
            ANCHORS_PATH,
        )

    def process_hand(self, frame, hand_data, hand_name):
        box = hand_data["box"]
        palm_points = hand_data["palm"]
        points = hand_data["joints"][:,:2]

        if box is not None:
            draw_box(frame, box)

        if palm_points is not None:
            draw_keypoints(frame, palm_points)
            pass

        if points is not None:
            if hand_name == "left":
                self.last_Lhandpoint = points
            if hand_name == "right":
                self.last_Rhandpoint = points

            # draw_keypoints(frame, points)
            # draw_connections(frame, points)


if __name__ == '__main__':
    init = main()
    cv2.namedWindow("Hand Tracking")
    capture = cv2.VideoCapture(0)
    if capture.isOpened():
        hasFrame, frame = capture.read()
    else:
        hasFrame = False

    while hasFrame:
        start = time.time()
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        init.detector(image)

        end = time.time()
        seconds = end - start
        fps = 1 / seconds

        if init.detector.left_hand["joints"] is not None:
            init.process_hand(frame, init.detector.left_hand, "left")
        else:
            left_history = []

        if init.detector.right_hand["joints"] is not None:
            init.process_hand(frame, init.detector.right_hand, "right")
        else:
            right_history = []

        fps_text = "fps : {0}".format(fps)
        draw_texts(frame, fps_text, (10,30))
        cv2.imshow("Hand Tracking", frame)

        hasFrame, frame = capture.read()

        key = cv2.waitKey(1)
        if key == 27:
            break

    capture.release()
    cv2.destroyAllWindows()