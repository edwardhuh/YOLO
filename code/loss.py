import tensorflow as tf
import numpy as np

from model.model import YOLO_Head

def compute_iou(pred_box:tf.Tensor, true_box:tf.Tensor):
    """
    Compute the IOU between two boxes
    """
    pred_box_xy = pred_box[..., 0:2]
    pred_box_wh = pred_box[..., 2:4]
    pred_box_min = pred_box_xy - pred_box_wh / 2
    pred_box_max = pred_box_xy + pred_box_wh / 2

    true_box_xy = true_box[..., 0:2]
    true_box_wh = true_box[..., 2:4]
    true_box_min = true_box_xy - true_box_wh / 2
    true_box_max = true_box_xy + true_box_wh / 2

    intersect_mins = tf.maximum(pred_box_min, true_box_min)
    intersect_max = tf.minimum(pred_box_max, true_box_max)
    intersect_wh = tf.maximum(intersect_max - intersect_mins, 0)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    pred_box_area = pred_box_wh[...,0] * pred_box_wh[...,1]
    true_box_area = true_box_wh[...,0] * true_box_wh[...,1]
    return intersect_area/(true_box_area + pred_box_area - intersect_area)    

def calc_scale(alpha, targets, preds, gamma):
    """
    Dynamically scale for confidence
    """
    return alpha * tf.pow(tf.abs(targets- tf.nn.sigmoid(preds)), gamma)

def compute_loss(y_pred:np.array, y_true:np.array, ignore_threshold:float=0.5, num_bounding_boxes=3):
    """
    Given y_pred, y_true, and additional model specifications, calculate the loss function
    args: 
        y_pred: an array of outputs of YOLO_Head.call(). [[box_xy, box_wh, box_confidence, box_class_probs], [box_xy, box_wh, box_confidence, box_class_probs], ...]
        y_true: an array of outputs of correct_ground_truths. [[box_xy, box_wh, box_confidence, box_class_probs], [box_xy, box_wh, box_confidence, box_class_probs], ...]
        num_bounding_boxes: defaults to 3 in kmeans.py implementation. 
    returns:
        loss: computed loss
    """
    loss_scale = []
    num_layers = 2 # default for tiny-yolo
    # We default to a 2 scale class. So, len(y_pred)==2
    assert len(y_pred) == 2

    GRID_SIZES = [(13,13), (26,26)]
    batch_size = tf.cast(len(y_true), tf.dtype(tf.float32)) # ensure to be float so that the resultant loss is float

    # initialize value
    overall_loss = 0

    # for loop for the loss, as each layer needs to be calculated separately. 
    # The associated layer and resultant grid traversed together. (ref `grid` variable)
    for l, grid_size in zip(range(num_layers), GRID_SIZES):
        # set up grid because XY Loss and WH Loss requires traversing the entire image
        grid_w = grid_size[0]
        grid_h = grid_size[1]
        grid = np.array([ [[float(x),float(y)]]*num_bounding_boxes   for y in range(grid_w) for x in range(grid_h)])

        # we need to get the xy values. This MIGHT not be index 3 depending on the input format
        pred_boxes = tf.reshape(y_pred[...,3:], (-1, grid_w*grid_h, num_bounding_boxes, 5))
        true_boxes = tf.reshape(y_true[...,3:], (-1, grid_w*grid_h, num_bounding_boxes, 5))
        y_true_conf = true_boxes[...,4]

        
    # Loss 1: XY Loss
        y_pred_xy = pred_boxes[...,0:2] + tf.variable(grid)
        y_true_xy = true_boxes[...,0:2]

        xy_loss = tf.sum(tf.sum(tf.square(y_true_xy - y_pred_xy), axis=-1)*y_true_conf, axis=-1)

    # Loss 2: WH Loss
        y_pred_wh = pred_boxes[...,2:4]
        y_true_wh = true_boxes[...,2:4]
        
        wh_loss = tf.sum(tf.sum(tf.square(tf.sqrt(y_true_wh) - tf.sqrt(y_pred_wh)), axis=-1)*y_true_conf, axis=-1) 

    # Loss 3: Confidence Loss (IOU)
        # find the predicted box
        box_xy, box_wh, pred_confidence, _ = y_pred[l]
        pred_box = tf.concat([box_xy, box_wh], axis=-1)
        # find the true box from the provided
        true_box_xy, true_box_wh, true_confidence, _ = y_true[l] 
        true_box = tf.concat([true_box_xy, true_box_wh], axis=-1)
        
        iou = compute_iou(pred_box=pred_box, true_box=true_box)
        conf_loss = tf.sum(tf.square(true_confidence*iou - pred_confidence)*true_confidence, axis=-1)

    # Loss 4: Class Loss 
    ## Because we are dealing with a single-class loss, classification loss is not necessary.
    ## source: https://stats.stackexchange.com/questions/312900/yolo-loss-function-for-detecting-1-class
    
    # Sum all loss
    # This is where potentially scaling can take place (i.e lambdas)
        loss = xy_loss + wh_loss + conf_loss
    
    overall_loss += loss
    return overall_loss

if __name__ == "__main__":
    compute_loss()