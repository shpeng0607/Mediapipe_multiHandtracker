import cv2
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import open3d

FILTER_COEFFS = [-0.17857, -0.07143, 0.03571, 0.14286, 0.25, 0.35714, 0.46429]

def rotation_matrix_from_vectors(vec1, vec2):
    """ 
    Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def find_finger_lengths(keypoints):
    fingers = [[i*4 + j +1 for j in range(4)] for i in range(5)]
    lengths = []
    for finger in fingers:
        f = finger[1:]
        f_shifted = finger[:-1]
        lengths.append(np.sqrt(((keypoints[f] - keypoints[f_shifted])**2).sum(1)).sum(0))
    return lengths

def normalize_keypoints(keypoints):
    palm_len = np.sqrt(((keypoints[9]-keypoints[0])**2).sum())
    norm_keypoints = (keypoints-keypoints[0]) / palm_len

    rot_mat = rotation_matrix_from_vectors(keypoints[9], np.array([0,-1,0]))
    rot_keypoints = norm_keypoints @ rot_mat
    return rot_keypoints, norm_keypoints

def orientation_keypoints(keypoints):
    middle_direction = (keypoints[2]-keypoints[0])[:2]
    angle = np.arctan2(-middle_direction[1], middle_direction[0]) * 180 / np.pi
    return angle

def plot_keypoints(keypoints):
    point_cloud = open3d.geometry.PointCloud()
    point_cloud.points = open3d.utility.Vector3dVector(keypoints)
    open3d.visualization.draw_geometries([point_cloud])

def smooth_keypoints(keypoints_list):
    smooth_points = (np.array(FILTER_COEFFS)[:,None] * np.array(keypoints_list)).sum(0)
    return smooth_points