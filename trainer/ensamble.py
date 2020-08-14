from ensemble_boxes import weighted_boxes_fusion
from utils import pascal2coco, coco2pascal

def run_wbf(boxes, scores, image_size=1024, iou_thr=0.55, skip_box_thr=0.1, weights=None):
    boxes = [coco2pascal(boxes)/image_size]
    labels = [np.ones(scores.shape[0]).tolist()]
    scores = [scores]
    
    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    boxes = boxes*(image_size-1)
    boxes = pascal2coco(boxes)
    
    return boxes, scores, labels