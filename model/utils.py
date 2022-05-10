import argparse
import re
from pathlib import Path

import numpy as np
import tensorflow as tf

GRID_SIZES = [13, 26]
anchor_boxes = [
    [
        [0.0263671875, 0.046153846153846156],
        [0.041666666666666664, 0.07505646217926168],
        [0.0732421875, 0.12987012987012986],
    ],
    [
        [0.004962779156327543, 0.007331378299120235],
        [0.009375, 0.014664711632453569],
        [0.015833333333333335, 0.026415094339622643],
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
    anchor_maxes = anchors / 2
    anchor_mins = -anchor_maxes
    box_wh = np.expand_dims(ground_truths[:, 2:4], axis=1)
    box_maxes = box_wh / 2
    box_mins = -box_maxes
    intersect_mins = np.maximum(box_mins, anchor_mins)
    intersect_maxes = np.minimum(box_maxes, anchor_maxes)
    intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.0)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    box_area = (ground_truths[:, 2] * ground_truths[:, 3]).reshape((-1, 1))
    anchor_area = anchors[..., 0] * anchors[..., 1]
    iou = intersect_area / (box_area + anchor_area - intersect_area + 1e-10)

    best_anchor = np.argmax(iou, axis=-1)

    # the downside of doing this is if there are multiple ground boxes in the same cell, the ground truth will be overwritten

    for j in range(n_bounding_boxes):
        grid_x = int(np.floor(ground_truths[j, 0] // cell_size))
        grid_y = int(np.floor(ground_truths[j, 1] // cell_size))

        output_tensor[grid_y, grid_x, best_anchor[j], :4] = ground_truths[
            j,
        ]
        output_tensor[grid_y, grid_x, best_anchor[j], 4:] = 1.0

    return tf.constant(output_tensor, dtype=tf.float32)


# if __name__ == "__main__":
#     ground_truths = tf.convert_to_tensor(
#         [[[0, 0, 0.5, 0.5], [1, 1, 0.5, 0.5]]], dtype=tf.float32
#     )

#     outputs = correct_ground_truths(ground_truths, GRID_SIZES, anchor_boxes)
#     assert outputs[0][0, 0, 0, 2, 4] == 1.0
#     assert outputs[1][0, 0, 0, 2, 2] == 0.5
#     assert outputs[0][0, 12, 12, 2, 4] == 1.0
#     assert outputs[1][0, 25, 25, 2, 2] == 0.5


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

    min_epoch = float("inf")
    max_epoch = 0
    min_epoch_file = ""
    max_epoch_file = ""
    num_weights = 0

    for weight_file in checkpoint_dir.glob("*.h5"):
        num_weights += 1
        file_epoch = int(weight_file.stem[9:12])
        if file_epoch > max_epoch:
            max_epoch = file_epoch
            max_epoch_file = weight_file.name
        if file_epoch < min_epoch:
            min_epoch = file_epoch
            min_epoch_file = weight_file.name

    return min_epoch_file, max_epoch_file, max_epoch, num_weights


def correct_boxes_and_scores(y_pred_list, score_threshold=0.6):
    """Correct bounding boxes for network output."""

    boxes_list = [[], []]
    scores_list = [[], []]

    for i, y_pred in enumerate(y_pred_list):
        box_xy, box_wh, box_confidence, box_class_probs = y_pred

        # correct boxes for network output
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]

        box_mins = box_yx - (box_hw / 2.0)
        box_maxes = box_yx + (box_hw / 2.0)

        boxes_tensor = tf.concat(
            [box_mins, box_maxes],
            axis=-1,
        )
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

    result_boxes_list = [
        tf.concat([boxes1, boxes2], axis=0) for boxes1, boxes2 in zip(*boxes_list)
    ]
    result_scores_list = [
        tf.concat([scores1, scores2], axis=0) for scores1, scores2 in zip(*scores_list)
    ]

    return result_boxes_list, result_scores_list


def non_max_suppression(
    boxes, scores, max_output_size=100, iou_threshhold=0.5, score_threshold=0.5
):
    """Perform non-max suppression on bounding boxes."""

    nms_index = tf.image.non_max_suppression(
        boxes,
        scores,
        max_output_size,
        iou_threshhold,
        score_threshold=score_threshold,
    )
    boxes = tf.gather(boxes, nms_index)
    scores = tf.gather(scores, nms_index)

    return boxes, scores


if __name__ == "__main__":
    print(scan_weight_files(Path("checkpoints/YOLOv3-050822-083257")))
