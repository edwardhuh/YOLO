# Object detection with YOLOv3

Our project will implement YOLO-v3, which is a single-shot object detection algorithm based on deep convolutional neural networks that detect and classify objects and predict bounding boxes within one pass of the image into the network. Our implementation will be based on the original YOLO paper (Redmon et al, 2016), which details the construction of the YOLO network, and the YOLO-v3 paper, which documents incremental changes to the YOLO network structure to achieve incremental improvement on the original. YOLO-v3 and its successors are still largely the state of the art algorithms in their efficiency in object detection over previous work such as R-CNN. By implementing this algorithm, we aim to achieve a better understanding of convolutional neural networks and how it achieves the same or better results than traditional computer vision algorithms in object detection.

## Setting up environment, and getting the crowdhuman data
Navigate to the `data` folder, then run the following command on the terminal: 
Note: this script assumes that you have the correct set up for `python3` and `pip`. 
```
bash get_data.sh
```
This script should automatically download the first 5000 image data from the Crowdhuman dataset in the `data/raw`, along with the relevant datasets in the correct directory structure.

## Anchor box generation through kmeans
We generate kmeans using 5,000 images of faces in the CrowdHuman dataset. The width and height of each image is normalized relative to the image (i.e goes from 0~1 as a proportion of 416). 

The outputs of the trained anchor sizes (with the default 6 anchor boxes as specified in Tiny YOLOv3) are availabe in `code/anchor_boxes.txt`. The breakdown of how the kmeans are created around different scales is available in `code/anchor_box_vis.png`.

## Training
After having the above set up, please run the following on your terminal:
```
python3 train.py
```
