import sys
sys.path.append("../")
import os
import cv2
from data import dataset
from utils import utils
from utils import vis_tools
from tqdm import tqdm
from config.config import cfg
from data import postprocess
from data import preprocess
from data import loader
import numpy as np


def get_idx(array):
	idx_tuple = np.where(array==1)
	u, idx = np.unique(idx_tuple[0], return_index = True)
	return u, idx_tuple[1][idx]


def parse_bevlabel(bevlabel, anchors):
	xmap = np.tile(np.array(range(cfg.BEV.OUTPUT_X))[:, np.newaxis], [1, cfg.BEV.OUTPUT_Y])
	ymap = np.tile(np.array(range(cfg.BEV.OUTPUT_Y))[np.newaxis, :], [cfg.BEV.OUTPUT_X, 1])
	xy_grid = np.stack((xmap,ymap), axis=-1)
	bevlabel = np.concatenate((bevlabel, xy_grid), axis=-1)
	labels = bevlabel[bevlabel[..., 0]==1]
	cls_type = labels[..., 1].astype(np.int32)
	prob = np.ones(cls_type.shape[0], dtype=np.float32)
	box = labels[..., 3:-2].reshape(-1, cfg.PIXOR.CLASSES_NUM, cfg.BEV.BBOX_DIM)
	box = box[np.arange(box.shape[0]), cls_type]
	xx = labels[..., -2] - box[..., 0] * anchors[cls_type,3] 
	yy = labels[..., -1] - box[..., 1] * anchors[cls_type,4]
	x = cfg.BEV.X_MAX - xx * cfg.BEV.X_RESOLUTION * cfg.BEV.STRIDE
	y = cfg.BEV.Y_MAX - yy * cfg.BEV.Y_RESOLUTION * cfg.BEV.STRIDE
	hwl = box[..., 2:5] * anchors[cls_type, :3]
	theta = np.arctan2(np.sin(box[..., 5]), np.cos(box[..., 5]))
	return np.stack([cls_type, prob, x, y, hwl[..., 0], hwl[..., 1], hwl[..., 2], theta], axis=-1)





trainset            = dataset.Dataset('train')
bev_anchors         = trainset.load_anchors(cfg.BEV.ANCHORS)

for _ in range(len(trainset)):
	data= trainset.load()
	bevlabel = parse_bevlabel(data[2][0], bev_anchors)
	bev_bboxes = postprocess.bev_nms(bevlabel, cfg.BEV.DISTANCE_THRESHOLDS)
	vis_tools.imshow_bev_bbox(data[0][0][..., -3:], np.array(bev_bboxes))
	pass
