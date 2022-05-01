from concurrent.futures import process
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf

import preprocess
from visualize_boxes import visualize


def main():
    if len(sys.argv) != 2:
        print("USAGE: python main.py <data_dir>")
        print("<data_dir> should be '../data' if running from main.py")
        exit()

    # Otherwise, we have data_dir
    data_dir = sys.argv[1]
    print(
        "WARNING: this script assumes that you have downloaded the Crowdhuman dataset using the provided get_data.sh script"
    )
    print("else, the script may error due to some hard coded file paths")
    if Path(data_dir).exists():
        ####
        # This should probably be a .config file.
        # But it gets the job done
        raw_data_dir = Path(data_dir) / "raw"
        raw_img_dir = raw_data_dir / "images"
        annotation_train_dir = raw_data_dir / "annotation_train.odgt"

        processed_data_dir = Path(data_dir) / "processed"
        resized_img_dir = processed_data_dir / "resized"
        resized_one_file = processed_data_dir / "resized_one_file"
        ####
        assert raw_img_dir.exists()

        x = preprocess.parse_data(
            raw_img_dir,
            annotation_train_dir,
            resized_img_dir,
            save_data=True,
        )
        y = preprocess.get_data(resized_img_dir / "annotations.json")
        assert x == y

        preprocess.parse_data_single(
            Path(data_dir) / "images/Images",
            Path(data_dir) / "annotation/annotation_train.odgt",
        )

        # Visualize a single image
        batch = preprocess.get_batch(y)
        # img = tf.keras.preprocessing.image.array_to_img(z)
        # plt.imshow(img)
        visualize(filepath=y[0]["file"], bounding_boxes=y[0]["boxes"], found_boxes=[])

    return None


if __name__ == "__main__":
    main()
