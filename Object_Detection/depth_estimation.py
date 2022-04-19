from homography import get_homography_matrix
from config.calibration import *


def estimate_depth(x, y):
    H = get_homography_matrix()
    H_inv = np.linalg.inv(H)
    vec = np.matmul(H_inv, np.array([x, y, 1]))
    depth = np.sqrt(np.power(vec[0], 2) + np.power(vec[1], 2)) / vec[2] / 100

    return depth
