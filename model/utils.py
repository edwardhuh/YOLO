import argparse
import os
import re
import numpy as np
import tensorflow as tf

from pathlib import Path

GRID_SIZES = [13, 26]
anchor_boxes = [
    [
        [216, 216],
        [16, 30],
        [33, 23],
    ],
    [
        [30, 61],
        [62, 45],
        [59, 119],
    ],
]


def correct_ground_truths(
    ground_truths, grid_sizes=GRID_SIZES, anchor_boxes=anchor_boxes
):

    output_tensors = []
    for grid_size, anchors in zip(grid_sizes, anchor_boxes):
        output_tensors.append(
            tf.map_fn(
                lambda x: correct_ground_truth(x, grid_size, anchors),
                ground_truths,
            )
        )

    return output_tensors


def correct_ground_truth(ground_truths, grid_size, anchor_boxes):
    """
    Processes the ground truths into the same shape as the outputs of YOLO head

    ground_truths: a (b, 4) tensor with all the correct bounding box information
        shaped as (bx, by, bw, bh)

        b: the number of true bounding boxes
        bx, by: the bounding box center coordinates of values between (0, 1) relative to
            the resized image (typically 416)
        bw, bh: the bounding box width and height of values between (0, 1) relative to
            the resized image

    grid_sizes: a list of the grid sizes at different scales of the model outputs

    returns:
        a list of tensors of shapes (m, grid_y, grid_x, n_anchors, 6), each corresponding to a different scale
    """

    ground_truths = ground_truths.numpy()
    ground_truths = ground_truths[ground_truths.sum(axis=1) > 0, :]
    n_bounding_boxes = ground_truths.shape[0]
    n_anchors = len(anchor_boxes)
    anchors = np.array(anchor_boxes, dtype=np.float32).reshape(-1, n_anchors, 2)

    output_tensor = np.zeros([grid_size, grid_size, n_anchors, 6], dtype=np.float32)
    cell_size = 1.0 / grid_size

    anchors = np.repeat(anchors, n_bounding_boxes, axis=0)
    intersect_mins = np.maximum(np.expand_dims(ground_truths[:, 2:4], axis=1), anchors)
    intersect_maxes = np.minimum(np.expand_dims(ground_truths[:, 2:4], axis=1), anchors)
    intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.0)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    box_area = (ground_truths[:, 2] * ground_truths[:, 3]).reshape((-1, 1))
    anchor_area = anchors[..., 0] * anchors[..., 1]
    iou = intersect_area / (box_area + anchor_area - intersect_area + 1e-10)

    best_anchor = np.argmax(iou, axis=-1)

    # the downside of doing this is if there are multiple ground boxes in the same cell, the ground truth will be overwritten

    for j in range(n_bounding_boxes):
        grid_x = int(ground_truths[j, 0] // cell_size)
        grid_y = int(ground_truths[j, 1] // cell_size)

        output_tensor[grid_y, grid_x, best_anchor[j], :4] = ground_truths[
            j,
        ]
        output_tensor[grid_y, grid_x, best_anchor[j], 4:] = 1.0

    return tf.constant(output_tensor, dtype=tf.float32)


if __name__ == "__main__":
    ground_truths = tf.convert_to_tensor(
        [[[0, 0, 0.5, 0.5], [0, 0, 0, 0]]], dtype=tf.float32
    )

    outputs = correct_ground_truths(ground_truths, GRID_SIZES, anchor_boxes)
    assert outputs[0][0, 0, 0, 0, 4] == 1.0


def parse_args():
    """Perform command-line argument parsing."""

    parser = argparse.ArgumentParser(
        description="Let's train some neural nets!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--load-checkpoint",
        default=None,
        help="""Path to model checkpoint file (should end with the
        extension .h5). Checkpoints are automatically saved when you
        train your model. If you want to continue training from where
        you left off, this is how you would load your weights.""",
    )

    return parser.parse_args()


def scan_weight_files(checkpoint_dir):
    """Scans checkpoint directory to find current minimum and maximum
    accuracy weights files as well as the number of weights."""

    min_acc = float("inf")
    max_acc = 0
    min_acc_file = ""
    max_acc_file = ""
    num_weights = 0

    for weight_file in checkpoint_dir.glob("*.h5"):
        num_weights += 1
        file_acc = float(
            re.findall(r"[+-]?\d+\.\d+", weight_file.name.split("acc")[-1])[0]
        )
        if file_acc > max_acc:
            max_acc = file_acc
            max_acc_file = weight_file.name
        if file_acc < min_acc:
            min_acc = file_acc
            min_acc_file = weight_file.name

    return min_acc_file, max_acc_file, max_acc, num_weights


def correct_boxes_and_scores(
    y_pred_list, input_size=416, image_sizes=None, score_threshold=0.6
):
    """Correct bounding boxes for network output."""

    boxes_list = [[], []]
    scores_list = [[], []]

    for i, y_pred in enumerate(y_pred_list):
        box_xy, box_wh, box_confidence, box_class_probs = y_pred

        # correct boxes for network output
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = tf.constant([input_size, input_size], dtype=tf.float32)

        if image_sizes is not None:
            image_shape = tf.constant(image_sizes, dtype=tf.float32)
        else:
            image_shape = input_shape

        new_shape = tf.round(image_shape * tf.reduce_min(input_shape / image_shape))
        offset = (input_shape - new_shape) / 2.0 / input_shape
        scale = input_shape / new_shape
        box_yx = (box_yx - offset) * scale
        box_hw *= scale

        box_mins = box_yx - (box_hw / 2.0)
        box_maxes = box_yx + (box_hw / 2.0)

        boxes_tensor = tf.concat([box_mins, box_maxes], axis=-1)
        box_scores_tensor = box_confidence * box_class_probs

        for boxes, box_scores in zip(
            tf.unstack(boxes_tensor), tf.unstack(box_scores_tensor)
        ):
            boxes = tf.reshape(boxes, [-1, 4])
            box_scores = tf.reshape(box_scores, [-1, 1])

            boxes = tf.boolean_mask(
                boxes, tf.reshape(box_scores > score_threshold, (-1,))
            )
            box_scores = tf.boolean_mask(box_scores, box_scores > score_threshold)

            boxes_list[i].append(boxes)
            scores_list[i].append(box_scores)

    for i, (boxes, scores) in enumerate(zip(boxes_list, scores_list)):
        boxes_list[i] = tf.concat(boxes, axis=0)
        scores_list[i] = tf.concat(scores, axis=0)

    return boxes_list, scores_list


def non_max_suppression(boxes, scores, max_output_size=100, iou_threshhold=0.5):
    """Perform non-max suppression on bounding boxes."""

    nms_index = tf.image.non_max_suppression(
        boxes,
        scores,
        tf.constant(max_output_size, dtype=tf.int32),
        tf.constant(iou_threshhold, dtype=tf.float32),
    )
    boxes = tf.gather(boxes, nms_index)
    scores = tf.gather(scores, nms_index)

    return boxes, scores


if __name__ == "__main__":
    print(scan_weight_files(Path("./checkpoints/YOLOv3-050622-061555")))
