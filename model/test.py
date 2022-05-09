from PIL import Image

import cv2
import numpy as np
import tensorflow as tf

from model import YOLOv3_Tiny
from utils import anchor_boxes, correct_boxes_and_scores, non_max_suppression


def scale_bbox_to_original(bboxes, original_hw):
    """
    scale bounding box to original size
    """
    original_hw = np.array(original_hw)

    box_xy_min = bboxes[:, :2] * original_hw
    box_xy_max = bboxes[:, 2:] * original_hw

    bboxes = np.concatenate([box_xy_min, box_xy_max], axis=1)

    return np.round(bboxes).astype(np.int32)


model = YOLOv3_Tiny(
    input_size=416,
    anchor_boxes=anchor_boxes,
    n_classes=1,
    iou_threshold=0.5,
    score_threshold=0.5,
)

model(tf.keras.Input(shape=(416, 416, 3)))
model.load_weights("./checkpoints/YOLOv3-050822-114021/weights.e049-acc0.3037.h5")

# load an image

original_image = Image.open("test1.jpg")
original_size = original_image.size
test_image = original_image.resize((416, 416))
original_image = np.array(original_image)
test_image = np.repeat(np.expand_dims(np.array(test_image) / 255.0, axis=0), 3, axis=0)

yolo_output = model(tf.constant(test_image), train=False)
boxes, scores = correct_boxes_and_scores(yolo_output)

boxes = boxes[0]
scores = scores[0]

pred_boxes, scores = non_max_suppression(boxes, scores, 30, 0.9)

bboxes = pred_boxes.numpy()
bboxes = scale_bbox_to_original(bboxes, original_size)

for bbox in bboxes:
    x1, y1, x2, y2 = bbox

    original_image = cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
cv2.imshow("test", original_image)
cv2.waitKey(0)
