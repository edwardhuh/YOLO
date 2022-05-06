import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from utils import CustomModelSaver, parse_args

from model import YOLOv3_Tiny, YOLO_Head, anchor_boxes
from utils import correct_ground_truths, CustomModelSaver, parse_args
from loss import compute_loss


MAX_BB_NUM = 179
GRID_SIZES = [13, 26]


# function to process images
def process_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.cast(tf.image.decode_jpeg(img, channels=3), tf.float32) / 255.0
    return img


def process_txt(txt_path):
    print(txt_path)
    txt = tf.io.read_file(txt_path, "utf-8")
    txt = tf.strings.to_number(tf.strings.split(txt, ","), tf.float32)
    txt = tf.reshape(txt, (-1, 4))
    if tf.shape(txt)[0] < MAX_BB_NUM:
        txt = tf.concat([txt, tf.zeros((MAX_BB_NUM - tf.shape(txt)[0], 4))], axis=0)
    elif tf.shape(txt)[0] > MAX_BB_NUM:
        txt = tf.random.shuffle(txt)[:MAX_BB_NUM]
    return txt


# Build two datasets, one for training and one for validation
image_dataset_train = tf.data.Dataset.list_files(
    "../data/processed/resized_one_file/*_416.jpg", shuffle=False
).map(process_image)

label_dataset_train = tf.data.Dataset.list_files(
    "../data/processed/resized_one_file/*_416.txt", shuffle=False
).map(process_txt)


ds = tf.data.Dataset.zip((image_dataset_train, label_dataset_train))

ds_train = (
    ds.take(100)
    .shuffle(buffer_size=1000)
    .cache()
    .prefetch(buffer_size=tf.data.AUTOTUNE)
    .batch(16, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)
)

ds_val = (
    ds.skip(4000)
    .shuffle(buffer_size=1000)
    .cache()
    .prefetch(buffer_size=tf.data.AUTOTUNE)
    .batch(128, num_parallel_calls=tf.data.AUTOTUNE)
)


if __name__ == "__main__":

    ARGS = parse_args()

    time_now = datetime.now()
    timestamp = time_now.strftime("%m%d%y-%H%M%S")

    checkpoint_dir = Path("./checkpoints/YOLOv3" + "-" + timestamp)

    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir()

    callbacks_list = [
        tf.keras.callbacks.TensorBoard(
            log_dir="./logs/YOLOv3" + "-" + timestamp,
            histogram_freq=0,
        ),
        CustomModelSaver(checkpoint_dir, 5),
    ]

    # Create the model
    model = YOLOv3_Tiny(
        input_size=416,
        anchor_boxes=anchor_boxes,
        n_classes=1,
        iou_threshold=0.5,
        score_threshold=0.5,
    )

    yolo_head_1 = YOLO_Head(anchor_boxes[0], n_classes=1)
    yolo_head_2 = YOLO_Head(anchor_boxes[1], n_classes=1)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    ### training the model:

    for epoch in range(100):
        print(epoch)
        for imgs, boxes in ds_train:
            y_true = correct_ground_truths(boxes, GRID_SIZES, anchor_boxes)

            with tf.GradientTape() as tape:
                y_pred_1, y_pred_2 = model(imgs)
                head_1 = yolo_head_1(y_pred_1, input_shape=416, train=True)
                head_2 = yolo_head_2(y_pred_2, input_shape=416, train=True)
                y_pred = [head_1, head_2]
                loss = compute_loss(y_true, y_pred, anchor_boxes, 0.5, 0.5)
                print(loss.numpy())
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
