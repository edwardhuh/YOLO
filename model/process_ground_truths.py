import numpy as np
import tensorflow as tf


def correct_ground_truths(ground_truths, grid_sizes):
    """
    Processes the ground truths into the same shape as the outputs of YOLO head

    ground_truths: a (m, b, 4) tensor with all the correct bounding box information
        shaped as (bx, by, bw, bh)

        m: batch size
        b: the number of true bounding boxes
        bx, by: the bounding box center coordinates of values between (0, 1) relative to
            the resized image (typically 416)
        bw, bh: the bounding box width and height of values between (0, 1) relative to
            the resized image

    grid_sizes: a list of the grid sizes at different scales of the model outputs

    returns:
        a list of tensors of shapes (m, grid_y, grid_x, 6)
    """

    m = ground_truths.shape[0]

    outputs = []

    for grid_size in grid_sizes:

        output_tensor = np.zeros([m, grid_size, grid_size, 6], dtype=np.float32)
        cell_size = 1.0 / grid_size

        for i in range(m):
            for j in range(ground_truths.shape[1]):
                grid_x = int(ground_truths[i, j, 0] // cell_size)
                grid_y = int(ground_truths[i, j, 1] // cell_size)

                output_tensor[grid_y, grid_x, :] = ground_truths[i, j, :]
                output_tensor[grid_y, grid_x, 4:] = 1.0

        outputs.append(output_tensor)

    return outputs
