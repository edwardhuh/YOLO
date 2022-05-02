import tensorflow as tf
import numpy as np


def process_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.cast(tf.image.decode_jpeg(img, channels=3), tf.float32) / 255.0
    return img, img_path


def process_csv(csv_path):
    labels = tf.io.read_file(csv_path)
    labels = tf.strings.split(labels, '\r\n')
    return labels


image_dataset = tf.data.Dataset.list_files(
    "../data/processed/resized_one_file/*_416.jpg", shuffle=False
).map(process_image)

label_dataset = tf.data.Dataset.list_files(
    "../data/processed/resized_one_file/*_416.txt", shuffle=False
).map(process_csv)


# dataset = tf.data.Dataset.zip((image_dataset, label_dataset))

for record, label in image_dataset.take(2):
    print(record, np.loadtxt(
        label.numpy().decode("utf-8").replace(".jpg", ".txt"),
        dtype=np.float32,
        delimiter=","
        ))
