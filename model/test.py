from PIL import Image
from pathlib import Path

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

    box_xy_min = bboxes[:, [1, 0]] * original_hw
    box_xy_max = bboxes[:, [3, 2]] * original_hw

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
model.load_weights(".\checkpoints\YOLOv3-050822-113608\weights.e091-acc0.3354.h5")

# load an image

output_dir = Path("./output")
original_images, resized_images = [], []
for image_file in Path("tests").glob("test*.jpg"):
    original_image = Image.open(image_file)
    test_image = original_image.resize((416, 416))
    original_images.append(original_image)
    test_image = np.array(test_image) / 255.0
    resized_images.append(test_image)

resized_images = tf.convert_to_tensor(resized_images)


yolo_output = model(resized_images, train=False)
boxes, scores = correct_boxes_and_scores(yolo_output)

for i, (original_image, boxes_pred, scores_pred) in enumerate(
    zip(original_images, boxes, scores)
):

    original_size = original_image.size
    original_image = np.array(original_image)
    boxes_pred, scores_pred = non_max_suppression(
        boxes_pred, scores_pred, 30, 0.5, 0.65
    )

    bboxes = boxes_pred.numpy()
    bboxes = scale_bbox_to_original(bboxes, original_size)

    for bbox in bboxes:
        x1, y1, x2, y2 = bbox

        original_image = cv2.rectangle(
            original_image, (x1, y1), (x2, y2), (0, 255, 0), 2
        )

    original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"outputs/test{i+1}.jpg", original_image)
