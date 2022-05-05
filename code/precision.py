import tensorflow as tf
import numpy as np
import sklearn.metrics

from code.loss import compute_iou

def average_precision(true_boxes:tf.Tensor, pred_boxes:tf.Tensor, threshold=0.5):
    iou = compute_iou(pred_boxes, true_boxes)
    # hits = np.nonzero(iou >= threshold)  # number of true positives
    # total_positives = np.nonzero(iou > 0)
    # precision is true positives / total positives
    precision = hits / total_positives
    # recall is true positives over total number of actual boxes
    recall = hits / len(true_boxes)
    

