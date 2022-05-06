import sys
from concurrent.futures import process
from pathlib import Path

import matplotlib.pyplot as plt
import preprocess
import tensorflow as tf
from kmeans import get_kmeans
from visualize_boxes import visualize

# from model.utils import correct_ground_truths


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
        resized_one_file_dir = processed_data_dir / "resized_one_file"
        ####
        assert raw_img_dir.exists()

        # Step 1-1: resize all images to 416 * 416, store to `resize_img_dir`
        # preprocess.parse_data(
        #     raw_img_dir,
        #     annotation_train_dir,
        #     resized_img_dir,
        #     save_data=True,
        # )
        # Step 1-2: resize all images to 416 * 416, store inidividual bounding boxes
        preprocess.parse_data_single(
            raw_img_dir, annotation_train_dir, resized_one_file_dir
        )

        # Step 2: read in processed json data
        y = preprocess.get_data(resized_img_dir / "annotations.json")

        # Visualize a single image
        # batch = preprocess.get_batch(y)
        # img = tf.keras.preprocessing.image.array_to_img(z)
        # plt.imshow(img)
        # visualize(filepath=y[0]["file"], bounding_boxes=y[0]["boxes"], found_boxes=[])

    # Step 3 compute kmeans
    # Get KMeans (defaulting to 3 anchor boxes)
    anchor_boxes = get_kmeans(
        resized_one_file_dir=resized_one_file_dir,
        annotation_filename=annotation_train_dir,
    )
    anchor_boxes = [
        [216, 216],
        [16, 30],
        [33, 23],
        [30, 61],
        [62, 45],
        [59, 119],
    ]

    ## Apply loss function

    return None


if __name__ == "__main__":
    main()
