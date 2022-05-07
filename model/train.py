import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf

from model import YOLOv3_Tiny, anchor_boxes
from utils import (
    correct_ground_truths,
    parse_args,
    scan_weight_files,
    correct_boxes_and_scores,
    non_max_suppression,
)
from loss import compute_loss
from precision import precision


MAX_BB_NUM = 179
GRID_SIZES = [13, 26]


# function to process images
def process_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.cast(tf.image.decode_jpeg(img, channels=3), tf.float32) / 255.0
    return img


def process_txt(txt_path):
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
    ds.take(4000)
    .shuffle(buffer_size=1000)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
    .batch(16, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)
)

ds_val = (
    ds.skip(4000)
    .shuffle(buffer_size=1000)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
    .batch(64, num_parallel_calls=tf.data.AUTOTUNE)
)


if __name__ == "__main__":

    ARGS = parse_args()

    # Create the model
    model = YOLOv3_Tiny(
        input_size=416,
        anchor_boxes=anchor_boxes,
        n_classes=1,
        iou_threshold=0.5,
        score_threshold=0.5,
    )

    init_epoch = None

    if ARGS.load_checkpoint is not None:
        ARGS.load_checkpoint = Path(ARGS.load_checkpoint)
        if not ARGS.load_checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint {ARGS.load_checkpoint} not found")

        checkpoint_dir = ARGS.load_checkpoint.parent
        model.load_weights(ARGS.load_checkpoint)
        regex = r"(?:.+)(?:\.e)(\d+)(?:.+)(?:.h5)"
        init_epoch = int(re.match(regex, ARGS.load_checkpoint.name).group(1)) + 1
        timestamp = checkpoint_dir.stem[-13:]

    else:
        time_now = datetime.now()
        timestamp = time_now.strftime("%m%d%y-%H%M%S")
        checkpoint_dir = Path("./checkpoints/YOLOv3" + "-" + timestamp)
        if not checkpoint_dir.exists():
            checkpoint_dir.mkdir()

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    ### training the model:

    epochs = 100
    max_num_weights = 5

    for epoch in range(epochs):

        print(f"Epoch {epoch+1}/{epochs}")

        pbar = tf.keras.utils.Progbar(target=len(ds_train), width=30)
        metrics = {}

        # train the model

        steps = 1
        for imgs, boxes in ds_train:

            y_true = correct_ground_truths(boxes, GRID_SIZES, anchor_boxes)

            with tf.GradientTape() as tape:
                y_pred = model(imgs)
                loss, results = compute_loss(y_true, y_pred, anchor_boxes, 0.5, 0.5)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            metrics.update(results)
            pbar.update(steps, values=metrics.items(), finalize=False)

            steps += 1

        # validation:
        precisions = []
        for imgs, true_boxes in ds_val:
            y_pred = model(imgs, train=False)
            boxes_list, scores_list = correct_boxes_and_scores(y_pred)

            for i, (raw_boxes, raw_scores, ground_truth_boxes) in enumerate(
                zip(boxes_list, scores_list, tf.unstack(true_boxes))
            ):
                pred_boxes, scores = non_max_suppression(
                    raw_boxes, raw_scores, 100, 0.5
                )
                pred_boxes = tf.gather(
                    pred_boxes,
                    tf.constant([1, 0, 3, 2], dtype=tf.int32),
                    axis=1,
                )  # xmin, ymin, xmax, ymax
                scores_sorted_ind = tf.argsort(scores, direction="DESCENDING")
                pred_boxes = tf.gather(pred_boxes, scores_sorted_ind, axis=0)
                pred_scores = tf.gather(scores, scores_sorted_ind)

                ground_truth_boxes = tf.boolean_mask(
                    ground_truth_boxes, tf.reduce_sum(ground_truth_boxes, axis=1) > 0.0
                )

                true_pos, false_pos = precision(
                    true_boxes=ground_truth_boxes,
                    pred_boxes=pred_boxes,
                    preprocess_true=True,
                )
                precisions.append([true_pos, false_pos])

        precisions = np.array(precisions)
        tp = np.sum(precisions[:, 0])
        fp = np.sum(precisions[:, 1])
        ap = tp / (tp + fp + 1e-10)

        metrics.update({"AP": ap, "TP": tp, "FP": fp})

        pbar.update(steps, values=metrics.items(), finalize=True)

        min_acc_file, max_acc_file, max_acc, num_weights = scan_weight_files(
            checkpoint_dir
        )

        cur_acc = ap

        # Only save weights if test accuracy exceeds the previous best
        # weight file
        if cur_acc > max_acc:
            save_name = "weights.e{0:03d}-acc{1:.4f}.h5".format(epoch, cur_acc)

            model.save_weights(checkpoint_dir / save_name)

            # Ensure max_num_weights is not exceeded by removing
            # minimum weight
            if max_num_weights > 0 and num_weights + 1 > max_num_weights:
                os.remove(checkpoint_dir / min_acc_file)
