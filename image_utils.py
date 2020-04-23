import cv2
import numpy as np

from boxes_utils import convert_boxes


def _read_image_if_need(*args):
    """
    for each item read image if it is path

    :param args: list: images or image paths
    :return: list: images
    """
    images = []
    for image in args:
        if type(image) == str:
            image = cv2.imread(image)
        images.append(image)
    return images


def get_padding(image, scale_shape=(1280, 1920), **args):
    if len(image.shape) < 3:
        image = np.pad(image, ((0, scale_shape[0] - image.shape[0]), (0, scale_shape[1] - image.shape[1])), **args)
    else:
        image = np.pad(image, ((0, scale_shape[0] - image.shape[0]), (0, scale_shape[1] - image.shape[1]), (0, 0)),
                       **args)
    return image


def cut_rectangular_image(image, box):
    """
    cut boxing image from image

    :param image: np.array: image
    :param box: np.array: box in [x_min, y_min, x_max, y_max] format
    :return: np.array: image in this box
    """
    x_min, y_min, x_max, y_max = box
    return image[y_min:y_max, x_min:x_max]


def cut_rectangular_image_by_mask(image, mask, get_mask=False):
    """
    returns the least rectangular image (and corresponding mask if get_mask is True),
    that contain this object(where mask is not zero)

    :param image: np.array: image
    :param mask: np.array: binary mask
    :param get_mask: bool: if True, return also mask
    :return: np.array or (np.array, np.array): image in least rectangular box (and mask)
    """
    box = cv2.boundingRect(mask)
    box = convert_boxes([box], input_format='xywh', output_format='min_max')[0]
    image = cut_rectangular_image(image, box)
    if not get_mask:
        return image
    mask = cut_rectangular_image(mask, box)
    return image, mask


def rotate_image(image, angle, borderValue=(255, 255, 255)):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(image, M, (nW, nH), borderValue=borderValue)
