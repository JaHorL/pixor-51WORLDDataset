import os
import shutil
import numpy as np
import cv2
from utils import vis_tools
from utils import utils
from utils import math
from utils import transform
from config.config import cfg


def parse_bev_predmap(predmap, anchors):
    xmap = np.tile(np.array(range(cfg.BEV.OUTPUT_X))[:, np.newaxis], [1, cfg.BEV.OUTPUT_Y])
    ymap = np.tile(np.array(range(cfg.BEV.OUTPUT_Y))[np.newaxis, :], [cfg.BEV.OUTPUT_X, 1])
    xy_grid = np.stack((xmap,ymap), axis=-1)
    predmap = np.concatenate((predmap, xy_grid), axis=-1)
    preds = predmap[math.sigmoid(predmap[..., 0])>0.3]
    objness = math.sigmoid(preds[..., 0])[..., np.newaxis]
    clsness = math.sigmoid(preds[..., 1:cfg.PIXOR.CLASSES_NUM+1])
    box = preds[..., cfg.PIXOR.CLASSES_NUM+1:-2].reshape(-1, cfg.PIXOR.CLASSES_NUM, cfg.BEV.BBOX_DIM)
    prob = clsness * objness
    cls_max_prob = np.max(prob, axis=-1)
    cls_idx = np.argmax(prob, axis=-1)
    box = box[np.arange(box.shape[0]), cls_idx]
    xx = preds[..., -2] - box[..., 0] * anchors[cls_idx, 3]
    yy = preds[..., -1] - box[..., 1] * anchors[cls_idx, 4]
    x = cfg.BEV.X_MAX - xx * cfg.BEV.X_RESOLUTION * cfg.BEV.STRIDE
    y = cfg.BEV.Y_MAX - yy * cfg.BEV.Y_RESOLUTION * cfg.BEV.STRIDE
    hwl = box[..., 2:5] * anchors[cls_idx][..., :3]
    theta = np.arctan2(np.sin(box[..., 5]), np.cos(box[..., 5]))
    result = np.stack([cls_idx, cls_max_prob, x, y, hwl[..., 0], hwl[..., 1], hwl[..., 2], theta], axis=-1)
    return result[cls_max_prob>0.3]


def bev_nms(bboxes, thresholds):

    def is_close_center(bbox1, bbox2, threshold):
        xy = bbox2[..., 2:4] - bbox1[..., 2:4]
        distance = np.sqrt(xy[:,0]*xy[:,0]+ xy[:,1]*xy[:,1])
        return distance < threshold

    def merge_obj_bboxes(bboxes, cls_type):
        if(len(bboxes)==0): return []
        new_box = np.zeros(bboxes.shape[-1])
        new_box[0] = cls_type
        new_box[1] = np.mean(bboxes[..., 1])
        new_box[2:] = np.sum(bboxes[..., 2:]*bboxes[..., 1][..., np.newaxis], axis=0)
        sum_of_prob = np.sum(bboxes[..., 1])
        new_box[2:] = new_box[2:] / sum_of_prob
        area = new_box[5] * new_box[6]        
        if sum_of_prob / area < 0.5:
            return []
        return new_box
    
    classes_in_bev = list(set(bboxes[:, 0]))
    best_bboxes = []
    for cls_type in classes_in_bev:
        cls_mask = (bboxes[:, 0] == cls_type)
        cls_bboxes = bboxes[cls_mask]
        while len(cls_bboxes):
            max_ind = np.argmax(cls_bboxes[:, 1])
            sample_bbox = cls_bboxes[max_ind]
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind+1: ]])
            distance_mask = is_close_center(sample_bbox, cls_bboxes, thresholds[int(cls_type)])
            obj_bboxes = cls_bboxes[distance_mask]
            merged_bbox = merge_obj_bboxes(obj_bboxes, cls_type)
            if len(merged_bbox):
                best_bboxes.append(merged_bbox)
            cls_bboxes = cls_bboxes[np.logical_not(distance_mask)]
    return np.array(best_bboxes)


