"""
This script implements kmeans clustering to obtain the anchor boxes for YOLO.
The bounding boxes are currently given by tuples of the form (x0, y0,x1, y1), where (x0,y0) 
are the coordinates of the lower left corner and the x1, y1 are the coordinates of the upper right corner.

We need to extract the width and height from these coordinates, 
and normalize data with respect to image width and height.

However, after preprocessing, we will always have the same image size for all images; 416 * 416.

The standard Kmeans is proven to be insufficient, and we need a different distance metric in the form of 1-IOU(box,centroid)
"""

import json
from ast import Continue
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster
from sklearn.cluster import KMeans

import seaborn as sns
    
# hyperparameter determining the number of kmeans
# this corresponds to the number of anchor boxes in YOLOv3
K_MEANS_CLUSTERS = 3

# we will be dealing with images all the same size: 256 * 256


def graph_clusters(X):
    plt.figure(figsize=(10, 10))
    plt.scatter(X[:, 0], X[:, 1], alpha=0.1)
    plt.title("Clusters", fontsize=20)
    plt.xlabel("normalized width", fontsize=20)
    plt.ylabel("normalized height", fontsize=20)
    plt.show()


def iou(box, clusters):
    """
    :param box:      np.array of shape (2,) containing w and h
    :param clusters: np.array of shape (N cluster, 2)
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou = intersection / (box_area + cluster_area - intersection)

    return iou


def get_anchor_boxes(ID, resized_one_file_dir, BBOX_WHS):
    assert resized_one_file_dir is not None
    bbox_path = Path(resized_one_file_dir) / f"{ID}_416.txt"
    if bbox_path.exists():
        with open(bbox_path, "r") as bboxes:
            for bbox in bboxes.readlines():
                coord_list = bbox.splitlines()[0].split(",")
                w = float(coord_list[2])
                h = float(coord_list[3])
                assert w >= 0 and w <= 1
                assert h >= 0 and h <= 1
                BBOX_WHS.append((w, h))


def get_kmeans(
    resized_one_file_dir,
    annotation_filename="raw/annotation_val.odgt",
    cluster_num=6,
    dist=np.median,
    seed=1470,
):
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
    # This visualization helps us understand that the KMeans is primarily very narrow,
    # and it is often a problem of scale.
    # graph_clusters(X)

    rows = len(X)

    distances = np.empty((rows, cluster_num))
    last_cluster = np.zeros((rows,))

    np.random.seed(seed)
    # initialize the cluster centers to be k items
    clusters = X[np.random.choice(rows, cluster_num, replace=False)]

    while True:
        # Step 1: allocate each item to the closest cluster centers
        for icluster in range(cluster_num):
            distances[:, icluster] = 1 - iou(clusters[icluster], X)
        nearest_clusters = np.argmin(distances, axis = 1)

        if(last_cluster == nearest_clusters).all():
            break

        # Step 2: calculate the cluster centers as mean (or median) of all the cases in the clusters.
        for cluster in range(cluster_num):
            clusters[cluster] = dist(X[nearest_clusters == cluster], axis=0)
        last_cluster = nearest_clusters
    
    # Step 3: Visualization
    current_palette = list(sns.xkcd_rgb.values())
    WithinClusterMeanDist = np.mean(distances[np.arange(distances.shape[0]),nearest_clusters])

    # Code for generating kmeans cluster graph
    for icluster in np.unique(nearest_clusters):

        pick = nearest_clusters==icluster
        c = current_palette[icluster]
        plt.rc('font', size=8) 
        plt.plot(X[pick,0],X[pick,1],"p",
                color=c,
                alpha=0.5,label="cluster = {}, N = {:6.0f}".format(icluster,np.sum(pick)))
        # plt.text(clusters[icluster,0],
        #         clusters[icluster,1],
        #         "c{}".format(icluster),
        #         fontsize=20,color="red")
        plt.title("Clusters")
        plt.xlabel("width")
        plt.ylabel("height")
        plt.legend(title="Mean IoU = {:5.4f}".format(WithinClusterMeanDist))  
        plt.savefig('anchor_box_vis.png')
    return clusters, nearest_clusters, distances
