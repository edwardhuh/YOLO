"""
This script implements kmeans clustering to obtain the anchor boxes for YOLO.
The bounding boxes are currently given by tuples of the form (x0, y0,x1, y1), where (x0,y0) 
are the coordinates of the lower left corner and the x1, y1 are the coordinates of the upper right corner.

We need to extract the width and height from these coordinates, 
and normalize data with respect to image width and height.

However, after preprocessing, we will always have the same image size for all images; 256 * 256.
"""

import json
from ast import Continue
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import KMeans

# hyperparameter determining the number of kmeans
# this corresponds to the number of anchor boxes in YOLOv3
K_MEANS_CLUSTERS = 3

# we will be dealing with images all the same size: 256 * 256


def get_anchor_boxes(ID, resized_one_file_dir, BBOX_WHS):
    assert resized_one_file_dir is not None
    bbox_path = Path(resized_one_file_dir) / f"{ID}_256.txt"
    if bbox_path.exists():
        with open(bbox_path, "r") as bboxes:
            for bbox in bboxes.readlines():
                coord_list = bbox.splitlines()[0].split(",")
                w = coord_list[2]
                h = coord_list[3]
                BBOX_WHS.append((w, h))


def get_kmeans(resized_one_file_dir, annotation_filename="raw/annotation_val.odgt"):
    # initializing BBOX list for kmeans
    BBOX_WHS = []

    with open(annotation_filename, "r") as fanno:
        for raw_anno in fanno.readlines():
            anno = json.loads(raw_anno)
            ID = anno["ID"]
            print("Processing ID for kmeans: %s" % ID)
            get_anchor_boxes(
                ID=ID, resized_one_file_dir=resized_one_file_dir, BBOX_WHS=BBOX_WHS
            )

    X = np.array(BBOX_WHS)
    kmeans = KMeans(n_clusters=K_MEANS_CLUSTERS, random_state=0).fit(X)
    centers = kmeans.cluster_centers_
    centers = centers[centers[:, 0].argsort()]  # sort by bbox w
    return centers
