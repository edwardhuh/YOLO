"""
Fundamental process:
Step 1: Select the box with highest objectiveness score
Step 2: Then, comapre the IOU of this box with other boxes
Step 3: Remove the bounding boxes with IOU > 50%
Step 4: Move to the next highest objectness score
Step 5: repeat 2-4

There is an existing implementation in tf.images.non_max_suppression.
But we will try to implement our own.
"""
import numpy as np


def NMS(boxes, overlapThresh=0.4):
    # return an empty list, if no boxes given
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]  # x coordinate of the top-left corner
    y1 = boxes[:, 1]  # y coordinate of the top-left corner
    x2 = boxes[:, 2]  # x coordinate of the bottom-right corner
    y2 = boxes[:, 3]  # y coordinate of the bottom-right corner
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    areas = (x2 - x1 + 1) * (
        y2 - y1 + 1
    )  # We have a least a box of one pixel, therefore the +1
    indices = np.arange(len(x1))
    for i, box in enumerate(boxes):
        temp_indices = indices[indices != i]
        xx1 = np.maximum(box[0], boxes[temp_indices, 0])
        yy1 = np.maximum(box[1], boxes[temp_indices, 1])
        xx2 = np.minimum(box[2], boxes[temp_indices, 2])
        yy2 = np.minimum(box[3], boxes[temp_indices, 3])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / areas[temp_indices]
        if np.any(overlap) > overlapThresh:
            indices = indices[indices != i]
    return boxes[indices].astype(int)
