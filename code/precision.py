import numpy as np
from shapely.geometry import Polygon

def IoU(true_box, predicted_box):
    truth = Polygon([true_box[0], true_box[1], true_box[0] + true_box[2], true_box[1] + true_box[3]])
    prediction = Polygon([predicted_box[0], predicted_box[1], predicted_box[0] + predicted_box[2], predicted_box[1] + predicted_box[3]])
    intersection = truth.intersection(prediction).area
    union = truth.union(prediction).area
    return intersection / union

def mAP(true_boxes, predicted_boxes):
    counts = 0
    for true_box in true_boxes:
        iou = 0.0
        for predicted_box in predicted_boxes:
            iou_new = IoU(true_box, predicted_box)
            iou = iou_new if iou_new > iou else iou
        if iou >= 0.5:
            counts += 1
    return counts / len(true_boxes)