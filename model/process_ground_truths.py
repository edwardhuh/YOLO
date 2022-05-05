import numpy as np
import tensorflow as tf

GRID_SIZES = [13, 26]
anchor_boxes = [
    [216, 216],
    [16, 30],
    [33, 23],
    [30, 61],
    [62, 45],
    [59, 119],
]


def correct_ground_truths(
    ground_truths, anchor_boxes=anchor_boxes, grid_sizes=GRID_SIZES
):
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
        a list of tensors of shapes (grid_y, grid_x, n_anchors, 6), each corresponding to a different scale
    """

    n_bounding_boxes = ground_truths.shape[0]
    n_anchors = len(anchor_boxes) // 2
    anchor_boxes = np.array(anchor_boxes).reshape(-1, n_anchors, 2)

    outputs = []

    for grid_size, anchors in zip(grid_sizes, anchor_boxes):

        output_tensor = np.zeros([grid_size, grid_size, n_anchors, 6], dtype=np.float32)
        cell_size = 1.0 / grid_size

        anchors = np.tile(anchors / 416, (n_bounding_boxes, 1, 1))
        intersect_maxes = np.maximum(
            np.expand_dims(ground_truths[:, 2:4], axis=1), anchors
        )
        intersect_mins = np.minimum(
            np.expand_dims(ground_truths[:, 2:4], axis=1), anchors
        )
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

        outputs.append(output_tensor)

    return outputs


if __name__ == "__main__":
    ground_truths = np.array([[0.25, 0.25, 0.25, 0.25],[0, 0, 0.5, 0.5]])
    anchor_boxes = [
        [216, 216],
        [16, 30],
        [33, 23],
        [30, 61],
        [62, 45],
        [59, 119],
    ]

    outputs = correct_ground_truths(ground_truths, anchor_boxes)
    assert outputs[0][0, 0, 0, 4] == 0.0
