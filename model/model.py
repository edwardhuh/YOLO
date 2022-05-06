import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import (
    Conv2D,
    Concatenate,
    UpSampling2D,
    MaxPooling2D,
    BatchNormalization,
    LeakyReLU,
)

from utils import correct_ground_truths

GRID_SIZES = [13, 26]
anchor_boxes = [
    [
        [10, 14],
        [16, 30],
        [33, 23],
    ],
    [
        [30, 61],
        [62, 45],
        [59, 119],
    ],
]


class YOLO_Head(tf.keras.layers.Layer):
    """
    Implements the YOLOv3 head.
    """

    # 2 values for box center offsets(in x an y, relative to cell center),
    # 2 values box size scales (in x and y, relative to anchor dimensions),
    # 1 value for objectness score (between 0 and 1),
    # number-of-classes values for class score (between 0 and 1).

    def __init__(self, anchor_boxes, n_classes=1, **kwargs):
        """
        Initializes the YOLOv3 head.

        Parameters
        ----------
        anchor_boxes : a list of 3 anchor boxes, in absolute scale relative to 416
            (width and height)
        n_classes : number of classes to predict
        """

        super(YOLO_Head, self).__init__(**kwargs)

        self.anchor_boxes = anchor_boxes
        self.n_anchors = len(anchor_boxes)
        self.n_classes = n_classes

    def call(self, inputs, input_shape, train=True):
        """Reshapes `inputs` into final YOLO output format:

        # 2 values for box center offsets(in x an y, relative to cell center),
        # 2 values box size scales (in x and y, relative to anchor dimensions),
        # 1 value for objectness score (between 0 and 1),
        # number-of-classes values for class score (between 0 and 1).
        """

        # 1. Reshape to batch_size x grid_x x grid_y x n_anchors x 6

        anchors_tensor = tf.reshape(
            tf.constant(self.anchor_boxes, tf.float32), [1, 1, 1, self.n_anchors, 2]
        )
        grid_y, grid_x = inputs.shape[1:3]

        inputs = tf.reshape(
            inputs, (-1, grid_y, grid_x, self.n_anchors, 5 + self.n_classes)
        )

        # 2. Split to batch_size x grid_size x grid_size x num_anchors x 2

        grid_ys = tf.tile(tf.expand_dims(tf.range(grid_y), axis=1), [1, grid_x])
        grid_xs = tf.tile(tf.expand_dims(tf.range(grid_x), axis=0), [grid_y, 1])
        grid = tf.concat(
            [
                grid_xs[:, :, tf.newaxis, tf.newaxis],
                grid_ys[:, :, tf.newaxis, tf.newaxis],
            ],
            axis=-1,
        )
        grid = tf.cast(grid, tf.float32)

        # 3. Add center offset and scale with anchors

        # box center offsets
        box_xy = (tf.math.sigmoid(inputs[..., :2]) + grid) / tf.convert_to_tensor(
            [grid_y, grid_x], tf.float32
        )
        # box width and height
        box_wh = tf.math.exp(inputs[..., 2:4]) * (
            anchors_tensor / tf.constant(input_shape, tf.float32)
        )
        # objectness score
        box_confidence = tf.math.sigmoid(inputs[..., 4:5])
        # class scores
        box_class_probs = tf.math.sigmoid(inputs[..., 5:])

        if train:
            return grid, inputs, box_xy, box_wh
        return box_xy, box_wh, box_confidence, box_class_probs


class ConvUnit(tf.keras.layers.Layer):
    def __init__(
        self,
        filters,
        kernel_size,
        name,
        strides=1,
        padding="SAME",
        activation=None,
        use_bias=False,
        pool=True,
        pool_size=2,
        pool_strides=2,
        **kwargs
    ):
        super(ConvUnit, self).__init__(name=name)
        self.pool = pool
        self.conv = Conv2D(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            activation=LeakyReLU(alpha=0.1) if activation is None else activation,
            use_bias=use_bias,
            **kwargs
        )
        self.bn = BatchNormalization()
        if self.pool:
            self.max_pool = MaxPooling2D(pool_size, pool_strides, padding=padding)

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        if self.pool:
            x = self.max_pool(x)
        return x


class YOLOv3_Tiny(tf.keras.Model):
    def __init__(
        self,
        input_size,
        anchor_boxes,
        iou_threshold,
        score_threshold,
        n_classes=1,
    ):
        super(YOLOv3_Tiny, self).__init__()
        self.n_classes = n_classes
        self.anchor_boxes = anchor_boxes
        self.n_anchors = sum([len(anchors) for anchors in anchor_boxes])
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.input_size = input_size

        self.conv1 = ConvUnit(16, 3, "conv1")
        self.conv2 = ConvUnit(32, 3, "conv2")
        self.conv3 = ConvUnit(64, 3, "conv3")
        self.conv4 = ConvUnit(128, 3, "conv4")
        self.conv5 = ConvUnit(256, 3, "conv5", pool=False)

        self.pool = MaxPooling2D(2, 2, padding="SAME")
        self.conv6 = ConvUnit(512, 3, "conv6", pool_strides=1)
        self.conv7 = ConvUnit(1024, 3, "conv7", pool=False)
        self.conv8 = ConvUnit(256, 1, "conv8", pool=False)

        self.conv9 = ConvUnit(512, 3, "conv9", pool=False)
        self.conv10 = ConvUnit(
            (self.n_anchors // 2) * (5 + self.n_classes),
            1,
            "conv10",
            pool=False,
            activation="linear",
        )

        # self.yolo_head_1 = YOLO_Head(self.anchor_boxes[0], n_classes=self.n_classes)

        ######

        self.concat = Concatenate(axis=-1)

        self.conv11 = ConvUnit(128, 1, "conv11", pool=False)
        self.upsample1 = UpSampling2D(2)

        self.conv12 = ConvUnit(256, 3, "conv12", pool=False)
        self.conv13 = ConvUnit(
            (self.n_anchors // 2) * (5 + self.n_classes),
            1,
            "conv13",
            activation="linear",
            pool=False,
        )

        # self.yolo_head_2 = YOLO_Head(self.anchor_boxes[1], n_classes=self.n_classes)

        ######

    def __call__(self, inputs):

        x1 = self.conv1(inputs)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        x1 = self.conv4(x1)
        x1 = self.conv5(x1)

        x2 = self.pool(x1)
        x2 = self.conv6(x2)
        x2 = self.conv7(x2)
        x2 = self.conv8(x2)

        y1 = self.conv9(x2)
        y1 = self.conv10(y1)
        # y1 = self.yolo_head_1(y1, input_shape=self.input_size, train=train)

        ######

        x2 = self.conv11(x2)
        x2 = self.upsample1(x2)

        x3 = self.concat([x2, x1])
        y2 = self.conv12(x3)
        y2 = self.conv13(y2)
        # y2 = self.yolo_head_2(y2, input_shape=self.input_size, train=train)

        ######

        return y1, y2

    def train_step(self, data):

        imgs, boxes = data
        y_true = correct_ground_truths(boxes, GRID_SIZES, self.anchor_boxes)

        with tf.GradientTape() as tape:
            y_pred = self(imgs)
            loss = self.compiled_loss(y_true, y_pred, self.anchor_boxes, 0.5, 0.5)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss

    def test_step(self, data):

        imgs, boxes = data
        y_true = correct_ground_truths(boxes, GRID_SIZES, self.anchor_boxes)

        y_pred = self(imgs)
        loss = self.compiled_loss(y_true, y_pred)

        return loss


if __name__ == "__main__":

    model = YOLOv3_Tiny(anchor_boxes, 0.5, 0.5, 1)

    test = tf.random.uniform((1, 416, 416, 3))
    y1, y2 = model(test)

    print(y1[0].shape)
    print(y2[0].shape)
