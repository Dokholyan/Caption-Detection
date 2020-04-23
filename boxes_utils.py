import numpy as np


def _random2min_max(points):
    """
    convert box from 4 random points format to [x_min, y_min, x_max, y_max] format

    :param points: np.array: 4 points as (x, y)
    :return: np.array: box in [x_min, y_min, x_max, y_max] format
    """
    x_max = max([x for x, y in points])
    x_min = min([x for x, y in points])
    y_max = max([y for x, y in points])
    y_min = min([y for x, y in points])
    return np.array([x_min, y_min, x_max, y_max])


def _xywh2min_max(box):
    """
    convert box from [x, y, w, h] format to [x_min, y_min, x_max, y_max] format

    :param box: np.array: box as [x, y, w, h]
    :return: np.array: box as [x_min, y_min, x_max, y_max]
    """
    x, y, w, h = box
    return np.array([x, y, x+w, y+h])


def convert_boxes(boxes, input_format='4_points', output_format='min_max'):
    """
    convert boxes
    4_points means 1----2 format, where each point is (x, y)
                  |    |
                  4----3
    min_max means [x_min, y_min, x_max, y_max] format
    4_random means the same as 4_points but in other order
    xywh  means [x, y, w, h] format, where (x, y) is a left up point

    :param boxes: list: list of boxes
    :param input_format: str: input boxes format
    :param output_format: str: output boxes format
    :return: list: boxes in output format
    """
    if input_format == output_format:
        return boxes
    new_boxes = []
    if input_format == '4_points' and output_format == 'min_max':
        for box in boxes:
            x = [point[0] for point in box]
            y = [point[1] for point in box]
            new_box = [min(x), min(y), max(x), max(y)]
            new_boxes.append(new_box)
        return np.array(new_boxes)
    elif input_format == 'min_max' and output_format == '4_points':
        for box in boxes:
            x_min, y_min, x_max, y_max = box
            new_box = ((x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max))
            new_boxes.append(new_box)
        return np.array(new_boxes)
    elif input_format == '4_random' and output_format == '4_points':
        for box in boxes:
            points = sorted(box, key=lambda point: point[1])
            up_points = sorted(points[:2], key=lambda point: point[0])
            bottom_points = sorted(points[2:], key=lambda point: point[0])
            new_box = [up_points[0], up_points[1], bottom_points[1], bottom_points[0]]
            new_boxes.append(new_box)
        return np.array(new_boxes)
    elif input_format == '4_random' and output_format == 'min_max':
        # ToDo
        pass
    elif input_format == 'xywh' and output_format == '4_points':
        for box in boxes:
            x, y, w, h = box
            box = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
            new_boxes.append(box)
        return np.array(new_boxes)
    elif input_format == '4_random' and output_format == 'min_max':
        for box in boxes:
            box = _random2min_max(box)
            new_boxes.append(box)
        return np.array(new_boxes)
    elif input_format == 'xywh' and output_format == 'min_max':
        for box in boxes:
            box = _xywh2min_max(box)
            new_boxes.append(box)
        return np.array(new_boxes)
    else:
        assert "There are some problems with formats, please check input_format and output_format"


def unite_boxes(boxes, input_format='min_max'):
    """
    computes a box containing all the other boxes

    :param boxes: list: boxes
    :param input_format: str: boxes format
    :return: np.array: box as [x_min, y_min, x_max, y_max]
    """
    boxes = convert_boxes(boxes, input_format=input_format, output_format='min_max')
    x_max = max([x_max for x_min, y_min, x_max, y_max in boxes])
    x_min = min([x_min for x_min, y_min, x_max, y_max in boxes])
    y_max = max([y_max for x_min, y_min, x_max, y_max in boxes])
    y_min = min([y_min for x_min, y_min, x_max, y_max in boxes])
    return np.array([x_min, y_min, x_max, y_max])


def get_distance(point_1, point_2):
    """
    calculate euclidean distance between points

    :param point_1: (float, float): point coordinates (x, y)
    :param point_2: (float, float): point coordinates (x, y)
    :return: float: distance between points
    """
    result = ((point_1[0] - point_2[0]) ** 2 + (point_1[1] - point_2[1]) ** 2) ** 0.5
    return result
