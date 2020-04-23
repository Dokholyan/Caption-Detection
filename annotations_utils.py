import cv2
import numpy as np
import random
import string


def create_mask_from_points(sets_of_points, size):
    """
    # ToDo replace it!
    Fills mask of size=size with
    polygons from sets_of_points,
    each polygon has color=idx inside
    :param sets_of_points: list, where each element is list
    of points for polygon
    :param size: size of mask
    :return: mask
    """
    mask = np.zeros(size)
    for idx, points in enumerate(sets_of_points):
        points = np.array(points).astype(np.int32)
        mask = cv2.fillPoly(mask, [points], color=idx + 1)
    return mask.astype(np.uint8)


def get_random_string(string_len=20):
    """
    generate a random string of fixed length
    :param string_len: int: string length
    :return: str: random string
    """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(string_len))