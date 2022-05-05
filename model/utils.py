import numpy as np
import tensorflow as tf

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

    anchors = np.repeat(anchors / 416, n_bounding_boxes, axis=0)
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
    anchor_boxes = [
        [216, 216],
        [16, 30],
        [33, 23],
        [30, 61],
        [62, 45],
        [59, 119],
    ]

    outputs = correct_ground_truths(ground_truths, GRID_SIZES, anchor_boxes)
    assert outputs[0][0, 0, 0, 0, 4] == 1.0


class CustomModelSaver(tf.keras.callbacks.Callback):
    """Custom Keras callback for saving weights of networks."""

    def __init__(self, checkpoint_dir, max_num_weights=5):
        super(CustomModelSaver, self).__init__()

        self.checkpoint_dir = checkpoint_dir
        self.max_num_weights = max_num_weights

    def on_epoch_end(self, epoch, logs=None):
        """At epoch end, weights are saved to checkpoint directory."""

        min_acc_file, max_acc_file, max_acc, num_weights = self.scan_weight_files()

        cur_acc = logs["val_sparse_categorical_accuracy"]

        # Only save weights if test accuracy exceeds the previous best
        # weight file
        if cur_acc > max_acc:
            save_name = "weights.e{0:03d}-acc{1:.4f}.h5".format(epoch, cur_acc)

            self.model.save_weights(self.checkpoint_dir / save_name)

            # Ensure max_num_weights is not exceeded by removing
            # minimum weight
            if self.max_num_weights > 0 and num_weights + 1 > self.max_num_weights:
                os.remove(self.checkpoint_dir / min_acc_file)

    def scan_weight_files(self):
        """Scans checkpoint directory to find current minimum and maximum
        accuracy weights files as well as the number of weights."""

        min_acc = float("inf")
        max_acc = 0
        min_acc_file = ""
        max_acc_file = ""
        num_weights = 0

        for weight_file in self.checkpoint_dir.glob("*.h5"):
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
