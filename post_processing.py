import cv2
import numpy as np

from boxes_utils import convert_boxes


def channels_last2first(masks):
    return np.transpose(masks, axes=(2, 0, 1))


def channels_first2last(masks):
    return np.transpose(masks, axes=(1, 2, 0))


class MorphPostProcessing:
    """
    Post process using morphological operations to clean binary mask and convert connected areas to boxes
    """
    def __init__(self, kernel_size=3, opening_it=1, erosion_it=1, dilation_it=1, contour_filter=None, box_filter=None):
        """
        Note: contour_filter always apply before box_filter

        :param kernel_size: (int, int): kernel shape
        :param opening_it: int: number of opening operations
        :param erosion_it: int: number of erosion operations
        :param dilation_it: int: number of dilation operations
        :param contour_filter: function: contours filter, function, that takes contours and mask and return filtered
                               contours
        :param box_filter: function: boxes filter, function, that takes boxes, mask, and boxes format and return
                                     filtered boxes
        """
        self.kernel = np.ones(kernel_size, np.uint8)
        self.opening_it = opening_it
        self.erosion_it = erosion_it
        self.dilation_it = dilation_it
        self.contour_filter = contour_filter
        self.box_filter = box_filter

    def transform_mask(self, mask):
        """
        postprocess mask using morphological operations

        :param mask: np.array[H, W]: binary mask
        :return: np.array[H, W]: postprocess mask
        """
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=self.opening_it)
        erosion = cv2.erode(opening, self.kernel, iterations=self.erosion_it)

        opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, self.kernel, iterations=self.opening_it)
        dilation = cv2.dilate(opening, self.kernel, iterations=self.dilation_it)
        return dilation

    @staticmethod
    def _contour2box(contour, angle=True, output_format='4_points'):
        """
        convert contour to box

        :param contour: np.array[Nx2]: open cv contour
        :param angle: bool: if True the function return rotating boxes False - function return ordinary boxes
        :param output_format: str: output boxes format
        :return: np.array: box
        """
        if angle:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = convert_boxes([box], input_format='4_random', output_format=output_format)[0]
        else:
            box = cv2.boundingRect(contour)
            box = convert_boxes([box], input_format='xywh', output_format=output_format)[0]
        return box

    def get_contours(self, mask):
        """
        extract contours from binary mask

        :param mask: np.array: binary mask
        :return: list of np.array [n_points, 1, 2]: contours
        """
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if self.contour_filter is not None:
            contours = self.contour_filter(contours, mask)
        return contours

    def get_boxes(self, mask, angle=True, output_format='4_points'):
        """
        extract boxes from binary mask

        :param mask: np.array: binary mask
        :param angle: bool: if True the function return rotating boxes False - function return ordinary boxes
        :param output_format: str: output boxes format
        :return: list: list of boxes
        """
        contours = self.get_contours(mask)
        boxes = []
        for contour in contours:
            box = self._contour2box(contour, angle, output_format)
            boxes.append(box)
        boxes = np.array(boxes)
        if self.box_filter is not None:
            boxes = self.box_filter(boxes, mask, input_format=output_format)
        return boxes

    def _process_single_mask(self, mask, **params):
        """
        postprocess single mask

        :param mask: np.array[H, W]: binary mask
        :param params: params for get_boxes functions, like angle(bool), output_format(str), ets
        :return: list: list of boxes
        """
        mask = self.transform_mask(mask)
        boxes = self.get_boxes(mask=mask, **params)
        return boxes

    def _process_several_masks(self, masks, channel_first=True, **params):
        """
        postprocess several mask

        :param masks: np.array: binary mask [C, H, W] or [H, W, C]
        :param channel_first: bool: channel is the first axis or not
        :param params: params for get_boxes functions, like angle(bool), output_format(str), ets
        :return: list[list]: list of boxes for each mask
        """
        if not channel_first:
            masks = channels_last2first(masks)
        result = []
        for mask in masks:
            result.append(self._process_single_mask(mask, **params))
        return result

    def process(self, mask, channel_first=True, **params):
        """
        post process mask

        :param mask: np.array: binary mask
        :param channel_first: bool: (if mask consist several masks) channel is the first axis or not
        :param params: params for get_boxes functions, like angle(bool), output_format(str), ets
        :return: list: list of boxes for each mask
        """
        if len(mask.shape) > 2:
            return self._process_several_masks(mask, channel_first, **params)
        else:
            return self._process_single_mask(mask, **params)
