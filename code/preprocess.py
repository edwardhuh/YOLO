import json
import os
from PIL import Image
import tensorflow as tf

def get_batch(labels):
    imgs = []
    for label in labels:
        with Image.open(label) as image:
            imgs.append(tf.keras.preprocessing.image.img_to_array(image) / 255)
    return imgs

def get_data(labels_file_path):
    with open(labels_file_path, "r") as f:
        return json.load(f)

def parse_data(inputs_file_path, labels_file_path, num_examples=100, save_data=False):
    """
    Takes in file paths and returns the images and associated labels

    args:
        inputs_file_path: the path to the folder holding the images
        labels_file_path: the path to the folder holding associated labels
        num_examples: the number of images to process
    returns:
        a list containing dictionaries with filepath and bounding box for each 256x256 image 
    """
    # First, load in the labels (annotations). They come in json format
    with open(labels_file_path, 'r') as l:
        labels = [json.loads(line.strip()) for line in l.readlines()]

    output = []
    i = 0
    for label in labels:
        # We want to limit the number of examples used
        if i >= num_examples:
            break
        curr = {}
        curr["ID"] = label["ID"]
        original_file = os.path.join(inputs_file_path, label["ID"] + ".jpg")
        # new file name for edited file
        curr["file"] = os.path.join(inputs_file_path, "resized", label["ID"] + "_256.jpg")
        # resize image and get image shape
        try:
            image = Image.open(original_file)
        except FileNotFoundError:
            continue
        sz = image.size
        img = image.resize([256, 256])
        if save_data:
            img.save(curr["file"])
        boxes = []
        # go through each bounding box
        for bounding_box in label["gtboxes"]:
            # don't get data that's unsure or ignored or out-of-bounds
            if not bounding_box["head_attr"].keys() or bounding_box["head_attr"]["unsure"] or bounding_box["head_attr"]["ignore"] or invalid(bounding_box["hbox"], sz):
                continue
            # normalize the bounding box between 0 and 1
            x_0 = bounding_box["hbox"][0] / sz[0]  # how far from left to start
            x_1 = bounding_box["hbox"][1] / sz[1]  # how far down to start
            x_2 = (bounding_box["hbox"][2] + bounding_box["hbox"][0]) / sz[0]  # how far the right edge is
            x_3 = (bounding_box["hbox"][3] + bounding_box["hbox"][1]) / sz[1]  # how far the lower edge is
            boxes.append([x_0, x_1, x_2, x_3])
        curr["boxes"] = boxes
        output.append(curr)
        i += 1
    if save_data:
        with open(os.path.join(inputs_file_path, "resized", "annotations.json"), "w") as f:
            json.dump(output, f)
    return output

def invalid(box, sz):
    """
    Checks to see if the bounding box is not wholly within the picture
    args:
        box: the bounding box
        sz: the size of the picture
    returns:
        True if the bounding box has sections outside the picture; otherwise False
    """
    return box[0] < 0 or box[1] < 0 or box[0] + box[2] > sz[0] or box[1] + box[3] > sz[1]

if __name__ == "__main__":
    x = prepare_data("images/CrowdHuman_train01/Images", "annotation/annotation_train.odgt", save_data=True)
    y = get_data("images/CrowdHuman_train01/Images/resized/annotations.json")
    assert x == y
