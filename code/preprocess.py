import json
import os
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image


def get_batch(labels):
    """
    Gets the images for one batch of labels
    args:
        labels: the annotations for the batch
    returns:
        list of images corresponding to the labels, each (416x416x3)
    """
    imgs = []
    for label in labels:
        with Image.open(label["file"]) as image:
            imgs.append(tf.keras.preprocessing.image.img_to_array(image) / 255)
    return imgs


def get_data(labels_file_path):
    """
    Loads annotations
    args:
        labels_file_path: filepath for the labels
    returns:
        the annotations (list of dicts)
    """
    with open(labels_file_path, "r") as f:
        return json.load(f)


def parse_data(
    inputs_file_path: Union[str, Path],
    labels_file_path: Union[str, Path],
    resized_img_dir: Union[str, Path],
    testing: bool = True,
    # num_examples: int = 100,
    save_data: bool = False,
):
    """
    Takes in file paths and returns the images and associated labels

    args:
        inputs_file_path: the path to the folder holding the images
        labels_file_path: the path to the folder holding associated labels
        testing: use smaller sample when testing function
    returns:
        a list containing dictionaries with filepath and bounding box for each 416x416 image
    """
    # First, load in the labels (annotations). They come in json format
    with open(labels_file_path, "r") as l:
        labels = [json.loads(line.strip()) for line in l.readlines()]
    if testing:
        num_examples = 100
    else:
        # otherwise, we want to use all 5_000
        num_examples = 5_000

    output = []
    i = 0
    for label in labels:
        # We want to limit the number of examples used
        if i >= num_examples:
            break
        # create a dictionary for the current image
        curr = {}
        curr["ID"] = label["ID"]  # id of this example
        original_file = (Path(inputs_file_path) / f'{label["ID"]}.jpg').as_posix()
        if not Path(original_file).exists():
            # When the original file path does not exist, simply skip
            continue
        # new file name for edited file
        curr["file"] = (Path(resized_img_dir) / f'{label["ID"]}_416.jpg').as_posix()

        # resize image and get image shape
        try:
            image = Image.open(original_file)
        except FileNotFoundError:
            continue
        sz = image.size
        img = image.resize([416, 416])
        if save_data:
            img.save(curr["file"])  # save the cropped image
        boxes = []
        # go through each bounding box
        for bounding_box in label["gtboxes"]:
            # don't get data that's unsure or ignored or out-of-bounds
            if (
                not bounding_box["head_attr"].keys()
                or bounding_box["head_attr"]["unsure"]
                or bounding_box["head_attr"]["ignore"]
                or invalid(bounding_box["hbox"], sz)
            ):
                continue
            # normalize the bounding box between 0 and 1
            # just looking at head boxes, not body boxes
            x_0 = bounding_box["hbox"][0] + bounding_box["hbox"][2] // 2
            x_1 = bounding_box["hbox"][1] + bounding_box["hbox"][3] // 2

            x_0 = x_0 / sz[0]  # how far from left to start
            x_1 = x_1 / sz[1]  # how far down to start
            x_2 = bounding_box["hbox"][2] / sz[0]  # width of box
            x_3 = bounding_box["hbox"][3] / sz[1]  # height of box
            boxes.append([x_0, x_1, x_2, x_3])
        # add boxes to current dict and current dict to list of dicts
        curr["boxes"] = boxes
        output.append(curr)
        i += 1
    if save_data:  # save the new annotation file
        with open(Path(resized_img_dir) / "annotations.json", "w") as f:
            json.dump(output, f)
    return output


def parse_data_single(
    inputs_file_path, labels_file_path, resized_one_file, num_examples=100
):
    """
    Parses data to create one .txt file for each image. Also resizes images to 416x416
    args:
        inputs_file_path: The path to the input .jpg
        labels_file_path: The path to the annotations file
        num_examples: The number of examples to parse. Parses all examples if set to -1
    returns: None
    """
    # First, load in the labels (annotations). They come in json format
    with open(labels_file_path, "r") as l:
        labels = [json.loads(line.strip()) for line in l.readlines()]

    i = 0
    for label in labels:
        # We want to limit the number of examples used
        if i >= num_examples and not num_examples == -1:
            break
        # create a dictionary for the current image
        original_file = os.path.join(inputs_file_path, label["ID"] + ".jpg")
        # new file name for edited file
        newpath = os.path.join(resized_one_file, label["ID"] + "_416")
        # resize image and get image shape
        try:
            image = Image.open(original_file)
        except FileNotFoundError:
            continue
        sz = image.size
        img = image.resize([416, 416])
        if Path(newpath).parent.exists():
            img.save(newpath + ".jpg")  # save the cropped image
        boxes = []
        # go through each bounding box
        for bounding_box in label["gtboxes"]:
            # don't get data that's unsure or ignored or out-of-bounds
            if (
                not bounding_box["head_attr"].keys()
                or bounding_box["head_attr"]["unsure"]
                or bounding_box["head_attr"]["ignore"]
                or invalid(bounding_box["hbox"], sz)
            ):
                continue
            # normalize the bounding box between 0 and 1
            # just looking at head boxes, not body boxes
            # get center instead of corner
            x_0 = bounding_box["hbox"][0] + bounding_box["hbox"][2] // 2
            x_1 = bounding_box["hbox"][1] + bounding_box["hbox"][3] // 2
            x_0 = x_0 / sz[0]  # how far from left to start
            x_1 = x_1 / sz[1]  # how far down to start
            x_2 = bounding_box["hbox"][2] / sz[0]  # width of box
            x_3 = bounding_box["hbox"][3] / sz[1]  # height of box
            boxes.append([x_0, x_1, x_2, x_3])
        # add boxes to current dict and current dict to list of dicts
        with open(newpath + ".txt", "w") as f:
            for box in boxes:
                f.write(",".join(map(str, box)) + "\n")
        i += 1


def invalid(box, sz):
    """
    Checks to see if the bounding box is not wholly within the picture
    args:
        box: the bounding box
        sz: the size of the picture
    returns:
        True if the bounding box has sections outside the picture; otherwise False
    """
    return (
        box[0] < 0 or box[1] < 0 or box[0] + box[2] > sz[0] or box[1] + box[3] > sz[1]
    )
