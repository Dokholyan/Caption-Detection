import cv2
import numpy as np
from image_utils import _read_image_if_need, get_padding, cut_rectangular_image
from post_processing import MorphPostProcessing


def remove_background(image, mask, background_color=255, background_label=0):
    """
    change pixels out of mask to background_color

    :param image: np.array: image
    :param mask: np.array: binary mask
    :param background_color: int: color, background will changed to this color
    :param background_label: int: background label in mask
    :return: np.array: image without background
    """
    image = image.copy()
    image[np.where(mask == background_label)] = background_color
    return image


def extract_objects_by_mask(image, mask, background_color=255):
    """
    extract object images and corresponding masks
    This function returns this least rectangular images and masks, where mask is not null

    :param image: np.array or str: image or image path
    :param mask: np.array or str: binary mask or mask path
    :param background_color: int or None: color, background will changed to this color if it is not None
    :return: (list(np.array), list(np.array)): images and masks
    """
    image, mask = _read_image_if_need(image, mask)
    box_extractor = MorphPostProcessing()
    boxes = box_extractor.get_boxes(mask, angle=False, output_format='min_max')
    object_images = []
    object_masks = []

    for box in boxes:
        object_image = cut_rectangular_image(image, box)
        object_mask = cut_rectangular_image(mask, box)
        if background_color:
            object_image = remove_background(object_image, object_mask, background_color)
        object_images.append(object_image)
        object_masks.append(object_mask)
    return object_images, object_masks


def image2background(background, watermark, mask=None, point=(0, 0), alpha=0):
    """
    simple image blending

    :param background: np.array: the image where it is inserted
    :param watermark: np.array: image to insert
    :param mask: np.array: watermark mask
    :param point: (int, int): upper left corner of image insertion
    :param alpha: float: defines watermark transparency, should be between 0 and 1
    :return: np.array, np.array: blending image, corresponding mask
    """
    background, watermark, mask = _read_image_if_need(background, watermark, mask)
    background = background.copy()
    if mask is None:
        mask = np.ones(watermark.shape[:2])

    # check inputs
    _check_params(background, watermark, mask, point)

    new_mask = np.zeros(background.shape[:2])
    rows, cols = watermark.shape[:2]
    new_mask[point[0]:rows + point[0], point[1]:cols + point[1]] = mask

    background_over_watermark = background[point[0]:rows + point[0], point[1]:cols + point[1]]

    background_over_watermark = cv2.addWeighted(background_over_watermark, alpha, watermark, 1. - alpha, 0.)

    background[point[0]:rows + point[0], point[1]:cols + point[1]][np.where(mask != 0)] = \
        background_over_watermark[np.where(mask != 0)]
    return background, new_mask.astype(np.uint8)


def _get_point_limits(background, watermark):
    watermark_height, watermark_width = watermark.shape[:2]
    background_height, background_width = background.shape[:2]
    max_y_shift = background_height - watermark_height
    max_x_shift = background_width - watermark_width
    return max_y_shift, max_x_shift


def _get_random_point(background, watermark, mode='common'):
    max_y_shift, max_x_shift = _get_point_limits(background, watermark)
    if mode == 'common':
        y = np.random.randint(0, max_y_shift)
        x = np.random.randint(0, max_x_shift)
    if mode == 'border':
        if np.random.random() > 0.5:
            y = np.random.choice([0, max_y_shift])
            x = np.random.randint(0, max_x_shift)
        else:
            x = np.random.choice([0, max_x_shift])
            y = np.random.randint(0, max_y_shift)
    return y, x


def _check_point(background, watermark, point):
    """


    :param background: np.array:
    :param watermark: np.array:
    :param point: (int, int)
    :return: bool
    """
    y, x = point
    max_y_shift, max_x_shift = _get_point_limits(background, watermark)
    if (y <= max_y_shift) and (y >= 0) and (x <= max_x_shift) and (x >= 0):
        return True
    return False


def _check_params(background, watermark, mask, point):
    if not _check_point(background, watermark, point):
        raise ValueError(f'point:{point[0], point[1]} is invalid!')
    if np.max(mask) > 1 or np.min(mask) < 0:
        raise ValueError(f'mask must be between 0 and 1!')
    if watermark.shape[:2] != mask.shape[:2]:
        raise ValueError(f'watermark shape{watermark.shape} and mask shape{mask.shape} must be the same(in H,W axis)!')


def image2random_position_background(background, watermark, mask=None):
    """

    :param background: np.array: the image where it is inserted
    :param watermark: np.array: image to insert
    :param mask: np.array: watermark mask
    :return: np.array, np.array: blending image, corresponding mask
    """
    background, watermark, mask = _read_image_if_need(background, watermark, mask)
    point = _get_random_point(background, watermark)
    background, new_mask = image2background(background, watermark, mask, point)
    return background, new_mask.astype(np.uint8)


def seamless_clone_image2background(background, watermark, mask=None, point=(0, 0), flag=cv2.NORMAL_CLONE):
    new_mask = np.zeros(background.shape[:2])
    rows, cols = watermark.shape[:2]
    center = (point[1] + cols // 2, point[0] + rows // 2)
    new_mask[point[0]:rows + point[0], point[1]:cols + point[1]] = mask
    output = cv2.seamlessClone(watermark,
                               background,
                               mask * 255, center, flag)
    return output, new_mask.astype(np.uint8)


def laplac_blending(background, watermark, mask, point, num_levels=3):
    """

    :param background: np.array: the image where it is inserted
    :param watermark: np.array: image to insert
    :param mask: np.array: watermark mask
    :param point: (int, int): upper left corner of image insertion
    :param num_levels: int: how many pyramids will be used
    :return: np.array, np.array: blending image, corresponding mask
    """
    background, watermark, mask = _read_image_if_need(background, watermark, mask)
    # check inputs
    _check_params(background, watermark, mask, point)
    H, W = background.shape[:2]
    background = _check_shape(background, num_levels)

    new_watermark, new_mask = image2background(np.ones_like(background) * 255,
                                               watermark, mask, point=point)

    laplas_mask = new_mask.astype(np.float32)
    if len(watermark.shape) == 3:
        laplas_mask = np.stack([laplas_mask, laplas_mask, laplas_mask], axis=-1)

    res = Laplacian_Pyramid_Blending_with_mask(background,
                                               new_watermark,
                                               1 - laplas_mask,
                                               num_levels)
    laplas = res.clip(0, 255).astype(np.uint8)
    return laplas[:H, :W], new_mask[:H, :W].astype(np.uint8)


def _check_shape(background, num_levels):
    H, W = background.shape[:2]
    if H % (2 ** num_levels) != 0 or W % (2 ** num_levels) != 0:
        new_H = int(np.ceil(H / (2 ** num_levels)) * 2 ** num_levels)
        new_W = int(np.ceil(W / (2 ** num_levels)) * 2 ** num_levels)
        background = get_padding(background, (new_H, new_W), mode='symmetric')
    return background


def double_laplac_blending(background, watermark, mask, point, num_levels=3):
    """

    :param background: np.array or str: the image(or path) where it is inserted
    :param watermark: np.array or str: image(or path) to insert
    :param mask: np.array or str: watermark mask(or path)
    :param point: (int, int): upper left corner of image insertion
    :param num_levels: int: how many pyramids will be used
    :return: np.array, np.array: blending image, corresponding mask
    """
    background, watermark, mask = _read_image_if_need(background, watermark, mask)

    # check inputs
    _check_params(background, watermark, mask, point)
    H, W = background.shape[:2]
    background = _check_shape(background, num_levels)

    laplas, new_mask = laplac_blending(background, watermark, mask, point, num_levels)

    laplas_mask = new_mask.astype(np.float32)
    # if color image
    if len(watermark.shape) == 3:
        laplas_mask = np.stack([laplas_mask, laplas_mask, laplas_mask], axis=-1)

    laplas = Laplacian_Pyramid_Blending_with_mask(laplas, background, laplas_mask, num_levels)

    laplas = laplas.clip(0, 255).astype(np.uint8)
    return laplas[:H, :W], new_mask[:H, :W].astype(np.uint8)


def Laplacian_Pyramid_Blending_with_mask(A, B, m, num_levels=6):
    """
    ToDo refact this!
    :param A:
    :param B:
    :param m:
    :param num_levels:
    :return:
    """
    # assume mask is float32 [0,1]

    # generate Gaussian pyramid for A,B and mask
    GA = A.copy()
    GB = B.copy()
    GM = m.copy()
    gpA = [GA]
    gpB = [GB]
    gpM = [GM]
    for i in range(num_levels):
        GA = cv2.pyrDown(GA)
        GB = cv2.pyrDown(GB)
        GM = cv2.pyrDown(GM)
        gpA.append(np.float32(GA))
        gpB.append(np.float32(GB))
        gpM.append(np.float32(GM))

    # generate Laplacian Pyramids for A,B and masks
    lpA = [gpA[num_levels - 1]]  # the bottom of the Lap-pyr holds the last (smallest) Gauss level
    lpB = [gpB[num_levels - 1]]
    gpMr = [gpM[num_levels - 1]]
    for i in range(num_levels - 1, 0, -1):
        # Laplacian: subtarct upscaled version of lower level from current level
        # to get the high frequencies
        LA = np.subtract(gpA[i - 1], cv2.pyrUp(gpA[i]))
        LB = np.subtract(gpB[i - 1], cv2.pyrUp(gpB[i]))
        lpA.append(LA)
        lpB.append(LB)
        gpMr.append(gpM[i - 1])  # also reverse the masks

    # Now blend images according to mask in each level
    LS = []
    for la, lb, gm in zip(lpA, lpB, gpMr):
        ls = la * gm + lb * (1.0 - gm)
        LS.append(ls)

    # now reconstruct
    ls_ = LS[0]
    for i in range(1, num_levels):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(ls_, LS[i])

    return ls_
