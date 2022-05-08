import tensorflow as tf

from model import YOLOv3_Tiny
from utils import anchor_boxes

model = YOLOv3_Tiny(
    input_size=416,
    anchor_boxes=anchor_boxes,
    n_classes=1,
    iou_threshold=0.5,
    score_threshold=0.5,
)

model(tf.keras.Input(shape=(416, 416, 3)))

model.load_weights("./checkpoints/YOLOv3-050822-083257/weights.e030-acc0.0240.h5")
