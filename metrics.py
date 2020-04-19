import rtree.index

from shapely.geometry import Polygon
from shapely.ops import polygonize

from boxes_utils import convert_boxes


def _contours2polygons(contours):
    polygons = []
    for contour in contours:
        polygons.append(Polygon(contour[:, 0, :]))
    return polygons


def _boxes2polygons(boxes, box_format='4_points'):
    new_boxes = convert_boxes(boxes, input_format=box_format, output_format='4_points')
    polygons = []
    for box in new_boxes:
        polygons.append(Polygon(box))
    return polygons


def get_polygon_iou(a, b):
    """
    calculate inception over union score by two polygons

    :param a: shapely.geometry.polygon.Polygon: polygon
    :param b: shapely.geometry.polygon.Polygon: polygon
    :return: float: iou score
    """
    return a.intersection(b).area / a.union(b).area


def _check_and_fix_polygons(polygons):
    for idx, pred_polygon in enumerate(polygons):
        if not pred_polygon.is_valid:
            exterior = pred_polygon.exterior
            mls = exterior.intersection(exterior)
            polygons[idx] = next(polygonize(mls))
    return polygons


def get_detection_statistics_polygons(pred_polygons, true_polygons, threshold=0.5, check_polygons=False):
    """
    calculate basic object detection statistics: true positive false positive, false negative, return result as a dict

    :param pred_polygons: list of polygons: prediction polygons
    :param true_polygons: list of polygons: ground truth polygons
    :param threshold: float: iou threshold
    :param check_polygons: bool: if True, all polygons will be checked and if they invalid, try to to fix this.
                                 This can significantly reduce the speed of work, so if you are sure that your polygons
                                 are not self-intersecting then use the flag check_polygons=False
    :return: dict: dictionary with keys: true_positive, false_positive, false_negative statistics
    """
    assert type(pred_polygons[0]) == Polygon, 'pred_polygons must be a list of shapely.geometry.polygon.Polygon'
    assert type(true_polygons[0]) == Polygon, 'true_polygons must be a list of shapely.geometry.polygon.Polygon'

    if check_polygons:
        pred_polygons = _check_and_fix_polygons(pred_polygons)
        true_polygons = _check_and_fix_polygons(true_polygons)

    true_positive, false_positive, false_negative = 0, 0, 0
    if len(pred_polygons) == 0:
        return {'true_positive': 0, 'false_positive': 0, 'false_negative': len(true_polygons)}
    if len(true_polygons) == 0:
        return {'true_positive': 0, 'false_positive': len(pred_polygons), 'false_negative': 0}
    pred_tree = rtree.index.Index()
    for i, a in enumerate(pred_polygons):
        pred_tree.insert(i, a.bounds)

    for gt_polygon in true_polygons:
        best_iou = 0
        best_id = None
        for i in pred_tree.intersection(gt_polygon.bounds):
            pred_polygon = pred_polygons[i]
            iou = get_polygon_iou(pred_polygon, gt_polygon)
            if iou > threshold and best_iou < iou:
                best_iou = iou
                best_id = i
        if best_id is not None:
            coordinates = pred_polygons[best_id].bounds
            pred_tree.delete(best_id, coordinates)
            true_positive += 1

    false_positive = len(pred_polygons) - true_positive
    false_negative = len(true_polygons) - true_positive
    return {'true_positive': true_positive,
            'false_positive': false_positive,
            'false_negative': false_negative}


def get_detection_statistics(prediction, ground_truth, threshold=0.5, check_input=False, input_type='4_points'):
    """
    calculate basic object detection statistics: true positive false positive, false negative, return result as a dict

    :param prediction: list of contours or boxes: predictions
    :param ground_truth: list of contours or boxes: ground_truth
    :param threshold: float: iou threshold
    :param check_input: bool: if True, all inputs will be checked and if they invalid, try to to fix this.
                              This can significantly reduce the speed of work, so if you are sure that your inputs
                              are not self-intersecting then use the flag check_input=False
    :param input_type: str: input type, should be 'contour' if inputs is np.array [n_points, 1, 2] or any type of boxes:
                            '4_points', 'min_max', 'xywh', '4_random'
    :return: dict: dictionary with keys: true_positive, false_positive, false_negative statistics
    """
    if input_type == 'contour':
        prediction = _contours2polygons(prediction)
        ground_truth = _contours2polygons(ground_truth)
    else:
        prediction = _boxes2polygons(prediction)
        ground_truth = _boxes2polygons(ground_truth, box_format=input_type)

    return get_detection_statistics_polygons(prediction, ground_truth, threshold=threshold,
                                             check_polygons=check_input)


def calculate_detection_metrics(true_positive, false_positive, false_negative):
    """
    calculate object detection metrics: precision, recall, f_score, return result as a dict

    :param true_positive: int: true positive
    :param false_positive: int: false positive
    :param false_negative: int: false negative
    :return: (float, float, float): precision, recall, f_score
    """
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f_score = 2 * (precision * recall) / (precision + recall + 1e-10)
    return {'precision': precision,
            'recall': recall,
            'f_score': f_score}
