import rtree.index
import matplotlib.pyplot as plt

from descartes import PolygonPatch
from shapely.geometry import Polygon
from shapely.ops import polygonize

from boxes_utils import convert_boxes
from image_utils import figure2image


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
        prediction = _boxes2polygons(prediction, box_format=input_type)
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


def show_iou_polygons(pred_polygon, true_polygon, inplace=True):
    """
    show intersection between polygons

    :param pred_polygon: polygon: prediction polygon
    :param true_polygon:  polygon: ground truth polygon
    :param inplace: bool: show figure or return image
    :return: None or np.array: return image if inplace=False
    """
    pred_polygon = _check_and_fix_polygons([pred_polygon])[0]
    true_polygon = _check_and_fix_polygons([true_polygon])[0]
    blue_color = (0, 0, 1)
    red_color = (0, 0.5, 0.5)
    green_color = (0, 1, 0)
    iou = get_polygon_iou(pred_polygon, true_polygon)
    intersection = pred_polygon.intersection(true_polygon)
    pred_ = pred_polygon.difference(true_polygon)
    ground_truth_ = true_polygon.difference(pred_polygon)
    (minx, miny, maxx, maxy) = pred_polygon.union(true_polygon).bounds
    fig, ax = plt.subplots(dpi=90)

    weights = maxx - minx
    heights = maxy - miny

    ax.set_xlim(left=minx - weights / 10, right=maxx + weights / 10)
    ax.set_ylim(bottom=miny - heights / 10, top=maxy + heights / 10)

    try:
        patch_pred = PolygonPatch(pred_, fc=blue_color, ec=blue_color, alpha=0.4, label='prediction')
        ax.add_patch(patch_pred)
    except:
        pass
    try:
        patch_gt = PolygonPatch(ground_truth_, fc=green_color, ec=green_color, alpha=0.4, label='ground truth')
        ax.add_patch(patch_gt)
    except:
        pass
    try:
        patch_intersection = PolygonPatch(intersection, fc=red_color, ec=red_color,
                                          alpha=0.2, label='intersection')
        ax.add_patch(patch_intersection)
    except:
        pass
    plt.title(f'iou={round(iou, 3)}')
    plt.legend(loc="lower right")
    if not inplace:
        image = figure2image(fig)
        plt.close(fig)
        return image
    else:
        plt.show()


def show_iou(prediction, ground_truth, inplace=True, input_type='4_points'):
    """
    show intersection between prediction and ground_truth

    :param prediction: contour or box: predictions
    :param ground_truth: contour or box: ground_truth
    :param inplace: bool: show image or return it
    :param input_type: input type, should be 'contour' if inputs is np.array [n_points, 1, 2] or any type of boxes:
    '4_points', 'min_max', 'xywh', '4_random'
    :return: None or np.array: return image if inplace=False
    """
    if input_type == 'contour':
        prediction = _contours2polygons([prediction])[0]
        ground_truth = _contours2polygons([ground_truth])[0]
    else:
        prediction = _boxes2polygons([prediction], box_format=input_type)[0]
        ground_truth = _boxes2polygons([ground_truth], box_format=input_type)[0]

    return show_iou_polygons(prediction, ground_truth, inplace=inplace)
