import tensorflow as tf
import numpy as np

from loss import compute_iou


def precision(true_boxes: tf.Tensor, pred_boxes: tf.Tensor, threshold=0.5):
    """
    Compute average precision for given true and predicted boxes.
    """

    # Compute IoU
    iou = compute_iou(pred_boxes, true_boxes, preprocess_true=True)

    n_pred = tf.shape(pred_boxes)[0].numpy()
    n_true = tf.shape(true_boxes)[0].numpy()

    i = 0

    true_pos = 0
    false_pos = 0

    remaining = np.ones(n_true, dtype=bool)

    while i < n_pred and remaining.sum() > 0:

        row = iou[i]

        if np.all(row < threshold):
            false_pos += 1
            i += 1
            continue

        row[~remaining] = 0
        assignment = np.argmax(row)
        remaining[assignment] = False
        true_pos += 1
        i += 1

    false_pos += n_pred - i

    return true_pos, false_pos


if __name__ == "__main__":
    y_pred = tf.random.uniform(shape=[10, 4])
    y_true = tf.random.uniform(shape=[1, 4])

    true_pos, false_pos = precision(y_true, y_pred)
    print(true_pos, false_pos)
