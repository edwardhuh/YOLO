import tensorflow as tf
import numpy as np

from loss import compute_iou


def precision(
    true_boxes: tf.Tensor,
    pred_boxes: tf.Tensor,
    threshold=0.5,
    preprocess_pred=False,
    preprocess_true=False,
):
    """
    Compute average precision for given true and predicted boxes.
    """
    n_pred = tf.shape(pred_boxes)[0].numpy()
    n_true = tf.shape(true_boxes)[0].numpy()
    if n_pred == 0:
        return 0, 0

    true_pos = 0
    false_pos = 0

    detected = np.zeros(n_true, dtype=bool)

    for i in range(n_pred):

        iou = compute_iou(
            tf.expand_dims(pred_boxes[i], axis=0),
            true_boxes,
            preprocess_pred=preprocess_pred,
            preprocess_true=preprocess_true,
        )

        if tf.reduce_all(iou < threshold).numpy():
            false_pos += 1
            continue

        iou = iou.numpy()
        iou[detected] = 0.0

        if np.all(iou <= threshold):
            false_pos += 1
            continue

        max_iou_ind = np.argmax(iou)
        detected[max_iou_ind] = True

        true_pos += 1

        if np.all(detected):
            false_pos += n_pred - i - 1
            break

    return true_pos, false_pos


if __name__ == "__main__":
    y_pred = tf.convert_to_tensor(
        [
            [0.2, 0.2, 1.0, 1.0],
            [0.3, 0.3, 0.7, 0.7],
        ],
        dtype=tf.float32,
    )
    y_true = tf.convert_to_tensor(
        [[0.0, 0.0, 1.0, 1.0], [0.5, 0.5, 0.7, 0.7], [0.5, 0.5, 1.0, 1.0]],
        dtype=tf.float32,
    )

    true_pos, false_pos = precision(y_true, y_pred)
    print(true_pos, false_pos)
