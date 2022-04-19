import cv2
import numpy as np


def get_homography_matrix():
    points_src = np.zeros((8, 2))
    points_dst = np.zeros((8, 2))

    points_dst[0] = [125, 525]
    points_dst[1] = [385, 530]
    points_dst[2] = [700, 530]
    points_dst[3] = [967, 520]
    points_dst[4] = [295, 457]
    points_dst[5] = [459, 455]
    points_dst[6] = [630, 455]
    points_dst[7] = [790, 453]

    points_src[0] = [-386, 80]
    points_src[1] = [-130, 80]
    points_src[2] = [129, 80]
    points_src[3] = [369, 80]
    points_src[4] = [-390, 670]
    points_src[5] = [-130, 670]
    points_src[6] = [123, 670]
    points_src[7] = [366, 670]

    H, mask = cv2.findHomography(points_src, points_dst)

    return H
