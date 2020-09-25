import cv2
import numpy as np

POINT_COLOR = (0, 255, 0)
CONNECTION_COLOR = (255, 0, 0)
THICKNESS = 2

connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
    (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)
]

def draw_box(frame, box):
    box = box.astype(int)
    cv2.circle(frame, (int(box.mean(axis=0)[0]), int(box.mean(axis=0)[1])), 1 * 2, (0, 0, 255), 2)
    cv2.drawContours(frame, [box.astype(int)], 0, (0, 0, 255), 2)

def draw_keypoints(frame, keypoints):
    for point in keypoints:
        x, y = point
        cv2.circle(frame, (int(x), int(y)), THICKNESS * 2, POINT_COLOR, THICKNESS)

def draw_connections(frame, points):
    for connection in connections:
        x0, y0 = points[connection[0]]
        x1, y1 = points[connection[1]]
        cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), CONNECTION_COLOR, THICKNESS)