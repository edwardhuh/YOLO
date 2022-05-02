import tensorflow as tf
import numpy as np

from process_ground_truths import correct_ground_truths, anchor_boxes
from yolov3_tiny import YOLOv3_Tiny


def process_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.cast(tf.image.decode_jpeg(img, channels=3), tf.float32) / 255.0
    return img


image_dataset = tf.data.Dataset.list_files(
    "../data/processed/resized_one_file/*_416.jpg", shuffle=False
).map(process_image)

label_dataset = tf.data.Dataset.list_files(
    "../data/processed/resized_one_file/*_416.txt", shuffle=False
)

dataset = tf.data.Dataset.zip((image_dataset, label_dataset)).batch(20)

# for record, label in image_dataset.take(2):
#     print(
#         record,
#         np.loadtxt(
#             label.numpy().decode("utf-8").replace(".jpg", ".txt"),
#             dtype=np.float32,
#             delimiter=",",
#         ),
#     )

model = YOLOv3_Tiny(
    anchors=anchor_boxes, n_classes=1, iou_threshold=0.5, score_threshold=0.5
)

for image, label in dataset:
    # y1, y2 = model(image)
    y1s_true = []
    y2s_true = []

    for label in label.numpy():
        label = label.decode("utf-8")
        ground_truths = np.loadtxt(label, dtype=np.float32, delimiter=",").reshape(
            (-1, 4)
        )
        y1, y2 = correct_ground_truths(ground_truths)
        y1s_true.append(y1)
        y2s_true.append(y2)

    y1_true = tf.convert_to_tensor(y1s_true)
    y2_true = tf.convert_to_tensor(y2s_true)

    print(y1_true.shape)
    print(y2_true.shape)
    break
