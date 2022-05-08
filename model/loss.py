from cv2 import exp
import tensorflow as tf

from typing import List


def compute_iou(
    pred_box: tf.Tensor,
    true_box: tf.Tensor,
    expand_dims: bool = False,
    preprocess_pred: bool = False,
    preprocess_true: bool = False,
):
    """
    Compute the IOU between two boxes
    """

    if expand_dims:
        pred_box = tf.expand_dims(pred_box, axis=-2)

    pred_box_xy = pred_box[..., 0:2]
    pred_box_wh = pred_box[..., 2:4]
    if preprocess_pred:
        pred_box_min = pred_box_xy - pred_box_wh / 2
        pred_box_max = pred_box_xy + pred_box_wh / 2
    else:
        pred_box_min = pred_box_xy
        pred_box_max = pred_box_wh
        pred_box_wh = pred_box_max - pred_box_min

    if expand_dims:
        true_box = tf.expand_dims(true_box, axis=-0)

    true_box_xy = true_box[..., 0:2]
    true_box_wh = true_box[..., 2:4]
    if preprocess_true:
        true_box_min = true_box_xy - true_box_wh / 2
        true_box_max = true_box_xy + true_box_wh / 2
    else:
        true_box_min = true_box_xy
        true_box_max = true_box_wh
        true_box_wh = true_box_max - true_box_min

    intersect_mins = tf.maximum(pred_box_min, true_box_min)
    intersect_max = tf.minimum(pred_box_max, true_box_max)
    intersect_wh = tf.maximum(intersect_max - intersect_mins, 0)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    pred_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]
    true_box_area = true_box_wh[..., 0] * true_box_wh[..., 1]
    return intersect_area / (true_box_area + pred_box_area - intersect_area)


def calc_scale(alpha, targets, preds, gamma):
    """
    Dynamically scale for confidence
    """
    return alpha * tf.pow(tf.abs(targets - tf.nn.sigmoid(preds)), gamma)


def compute_ignore_mask(
    pred_box: tf.Tensor,
    true_box: tf.Tensor,
    object_mask: tf.Tensor,
    ignore_thresh: float,
):
    """
    Compute the ignore mask for the loss
    """
    concatenated = tf.concat([pred_box, true_box, object_mask], axis=-1)
    ignore_mask = tf.map_fn(
        lambda x: compute_ignore_mask_per_image(x, ignore_thresh),
        concatenated,
        dtype=tf.bool,
    )

    return tf.cast(tf.expand_dims(ignore_mask, -1), dtype=tf.float32)


def compute_ignore_mask_per_image(concatenated, ignore_thresh):

    pred_box = concatenated[..., 0:4]
    true_box = concatenated[..., 4:8]
    object_mask = concatenated[..., 8]

    object_mask_bool = tf.cast(object_mask, tf.bool)
    true_box = tf.boolean_mask(true_box, object_mask_bool)
    ious = compute_iou(pred_box, true_box, expand_dims=True)
    best_ious = tf.reduce_max(ious, axis=-1)
    ignore_mask_per_image = best_ious < ignore_thresh

    return ignore_mask_per_image


def logit(x):
    return tf.math.log(x / (1.0 - x + 1e-6))


def compute_loss(
    y_true_list: List[List[tf.Tensor]],
    y_pred_list: List[tf.Tensor],
    anchor_boxes: List[List[float]],
    iou_threshold: float = 0.5,
    score_threshold: float = 0.5,
):
    """
    Given y_pred, y_true, and additional model specifications, calculate the loss function
    args:
        y_pred_list: the output of the model, it is a length-2 list of the following:
            grid: the grid for computing raw xy coordinates
            raw_pred: the raw prediction, which is a tensor of shape (batch_size, grid_size, grid_size, num_anchors, 5 + num_classes)
            box_xy_pred: the predicted box center offsets, which is a tensor of shape (batch_size, grid_size, grid_size, num_anchors, 2), and xy is relative to the entire image. This is needed to calculate IOU
            box_wh_pred: the predicted box width and height, which is a tensor of shape (batch_size, grid_size, grid_size, num_anchors, 2), and wh is relative to the entire image. This is needed to calculate IOU

        y_true_list: an array of outputs of correct_ground_truths, it is a list of twotensors of dimension (batch_size, grid_size, grid_size, num_anchors, 5 + num_classes))
    returns:
        loss: computed loss
    """
    grid_sizes = [13, 26]
    num_layers = 2  # default for tiny-yolo
    # We default to a 2 scale class. So, len(y_pred)==2
    assert len(y_pred_list) == 2

    # initialize value
    overall_loss = 0

    # for loop for the loss, as each layer needs to be calculated separately.
    # The associated layer and resultant grid traversed together. (ref `grid` variable)
    results = {}
    for l, grid_size in enumerate(grid_sizes):

        grid, y_pred_raw, box_xy_head, box_wh_head = y_pred_list[l]
        y_true = y_true_list[l]
        anchor_boxes_tensor = tf.reshape(
            tf.convert_to_tensor(anchor_boxes[l], dtype=tf.float32), [-1, 3, 2]
        )
        object_mask = y_true[..., 4:5]
        grid = grid * object_mask

        batch_size = tf.cast(tf.shape(y_pred_raw)[0], tf.float32)

        # we need to get the xy values. This MIGHT not be index 3 depending on the input format
        y_pred_conf = y_pred_raw[..., 4:5]
        box_loss_scale = 2 - y_true[l][..., 2:3] * y_true[l][..., 3:4]

        # Loss 1: XY Loss

        # Compute raw xy:
        box_xy_true = y_true[..., 0:2] * grid_size - grid
        box_xy_pred = tf.nn.sigmoid(y_pred_raw[..., 0:2])

        xy_loss = (
            tf.reduce_sum(
                tf.keras.losses.mean_squared_error(box_xy_true, box_xy_pred)
                * object_mask[..., 0]
                * box_loss_scale[..., 0]
            )
            / batch_size
        ) * 5

        # Loss 2: WH Loss
        box_wh_true = tf.math.log(y_true[..., 2:4] / anchor_boxes_tensor)
        box_wh_true = tf.keras.backend.switch(
            object_mask, box_wh_true, tf.zeros_like(box_wh_true)
        )
        box_wh_pred = y_pred_raw[..., 2:4]

        wh_loss = (
            tf.reduce_sum(
                tf.square(box_wh_true - box_wh_pred) * object_mask * box_loss_scale
            )
            / batch_size
        ) * 0.5

        # Loss 3: Confidence Loss (IOU)
        # find the predicted box
        bb_pred = tf.concat([box_xy_head, box_wh_head], axis=-1)
        bb_true = y_true[..., 0:4]

        ignore_mask = compute_ignore_mask(bb_pred, bb_true, object_mask, iou_threshold)

        bce_loss = tf.keras.losses.binary_crossentropy(
            object_mask, y_pred_conf, from_logits=True
        )
        conf_loss = (
            tf.reduce_sum(object_mask[..., 0] * bce_loss)
            + 0.5
            * tf.reduce_sum((1 - object_mask[..., 0]) * bce_loss * ignore_mask[..., 0])
        ) / batch_size

        # Loss 4: Class Loss
        class_loss = (
            tf.keras.losses.binary_crossentropy(
                y_true[..., 5:], y_pred_raw[..., 5:], from_logits=True
            )
            * object_mask[..., 0]
        )
        class_loss = tf.reduce_sum(class_loss) / batch_size

        # Loss 5: Total Loss
        total_loss = xy_loss + wh_loss + conf_loss # + class_loss

        results[f"xy_{l}"] = xy_loss.numpy()
        results[f"wh_{l}"] = wh_loss.numpy()
        results[f"cf_{l}"] = conf_loss.numpy()
        # results[f"cl_{l}"] = class_loss.numpy() not showing class loss because we know it's going to be very small

        overall_loss += total_loss
    results["loss"] = overall_loss.numpy()

    return overall_loss, results


if __name__ == "__main__":
    y_pred = tf.ones(shape=[2, 4])
    y_true = tf.ones(shape=[1, 4]) / 2
    print(compute_iou(y_pred, y_true))
