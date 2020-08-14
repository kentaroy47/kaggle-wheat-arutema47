import pandas as pd
import numpy as np
import numba
import re
import cv2
import ast
import matplotlib.pyplot as plt

from numba import jit
from typing import List, Union, Tuple

from collections import namedtuple
from typing import List, Union

Box = namedtuple('Box', 'xmin ymin xmax ymax')


def calculate_iou(gt: List[Union[int, float]],
                  pred: List[Union[int, float]],
                  form: str = 'pascal_voc') -> float:
    """Calculates the IoU.
    
    Args:
        gt: List[Union[int, float]] coordinates of the ground-truth box
        pred: List[Union[int, float]] coordinates of the prdected box
        form: str gt/pred coordinates format
            - pascal_voc: [xmin, ymin, xmax, ymax]
            - coco: [xmin, ymin, w, h]
    Returns:
        IoU: float Intersection over union (0.0 <= iou <= 1.0)
    """
    if form == 'coco':
        bgt = Box(gt[0], gt[1], gt[0] + gt[2], gt[1] + gt[3])
        bpr = Box(pred[0], pred[1], pred[0] + pred[2], pred[1] + pred[3])
    else:
        bgt = Box(gt[0], gt[1], gt[2], gt[3])
        bpr = Box(pred[0], pred[1], pred[2], pred[3])
        

    overlap_area = 0.0
    union_area = 0.0

    # Calculate overlap area
    dx = min(bgt.xmax, bpr.xmax) - max(bgt.xmin, bpr.xmin)
    dy = min(bgt.ymax, bpr.ymax) - max(bgt.ymin, bpr.ymin)

    if (dx > 0) and (dy > 0):
        overlap_area = dx * dy

    # Calculate union area
    union_area = (
            (bgt.xmax - bgt.xmin) * (bgt.ymax - bgt.ymin) +
            (bpr.xmax - bpr.xmin) * (bpr.ymax - bpr.ymin) -
            overlap_area
    )

    return overlap_area / union_area



def find_best_match(gts, predd, threshold=0.5, form='pascal_voc'):
    """Returns the index of the 'best match' between the
    ground-truth boxes and the prediction. The 'best match'
    is the highest IoU. (0.0 IoUs are ignored).
    
    Args:
        gts: Coordinates of the available ground-truth boxes
        pred: Coordinates of the predicted box
        threshold: Threshold
        form: Format of the coordinates
        
    Return:
        Index of the best match GT box (-1 if no match above threshold)
    """
    best_match_iou = -np.inf
    best_match_idx = -1
    
    for gt_idx, ggt in enumerate(gts):
        iou = calculate_iou(ggt, predd, form=form)
        
        if iou < threshold:
            continue
        
        if iou > best_match_iou:
            best_match_iou = iou
            best_match_idx = gt_idx

    return best_match_idx


def calculate_precision(preds_sorted, gt_boxes, threshold=0.5, form='coco'):
    """Calculates precision per at one threshold.
    
    Args:
        preds_sorted: 
    """
    tp = 0
    fp = 0
    fn = 0

    fn_boxes = []

    for pred_idx, pred in enumerate(preds_sorted):
        best_match_gt_idx = find_best_match(gt_boxes, pred, threshold=threshold, form='coco')

        if best_match_gt_idx >= 0:
            # True positive: The predicted box matches a gt box with an IoU above the threshold.
            tp += 1

            # Remove the matched GT box
            gt_boxes = np.delete(gt_boxes, best_match_gt_idx, axis=0)

        else:
            # No match
            # False positive: indicates a predicted box had no associated gt box.
            fn += 1
            fn_boxes.append(pred)

    # False negative: indicates a gt box had no associated predicted box.
    fp = len(gt_boxes)
    precision = tp / (tp + fp + fn)
    return precision, fn_boxes, gt_boxes


def calculate_image_precision(preds_sorted, gt_boxes, thresholds=(0.5), form='coco', debug=False):
    
    n_threshold = len(thresholds)
    image_precision = 0.0
    
    for threshold in thresholds:
        precision_at_threshold, _, _ = calculate_precision(preds_sorted,
                                                           gt_boxes,
                                                           threshold=threshold,
                                                           form=form
                                                          )
        if debug:
            print("@{0:.2f} = {1:.4f}".format(threshold, precision_at_threshold))

        image_precision += precision_at_threshold / n_threshold
    
    return image_precision

