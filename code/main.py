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
    if Path(data_dir).exists():
        # ensure that all the necessary file structures exist
        Path(data_dir/"images/Images/resized").mkdir(exist_ok=True)
        Path(data_dir/"images/Images/resized_one_file").mkdir(exist_ok=True)

        x = preprocess.parse_data(
            Path(data_dir) / "images/Images",
            Path(data_dir) / "annotation/annotation_train.odgt",
            save_data=True,
        )
        y = preprocess.get_data(
            Path(data_dir) / "images/Images/resized/annotations.json"
        )
        assert x == y


        
        preprocess.parse_data_single(
            Path(data_dir) / "images/Images",
            Path(data_dir) / "annotation/annotation_train.odgt",
        )

        # Visualize a single image
        batch = preprocess.get_batch(y)
        # img = tf.keras.preprocessing.image.array_to_img(z)
        # plt.imshow(img)
        visualize(filepath=y[0]['file'], bounding_boxes=y[0]['boxes'], found_boxes=[])
        
    return None


if __name__ == "__main__":
    main()
