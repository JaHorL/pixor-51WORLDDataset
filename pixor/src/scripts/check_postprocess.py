import sys
sys.path.append("../")
import os
import cv2
from glob import glob
from utils import math
from utils import vis_tools
from config.config import cfg
from data import postprocess
from data import loader
import numpy as np  


bev_pred_files      = glob(cfg.PIXOR.LOG_DIR+"/pred/bev_pred/*")
bev_anchors         = loader.load_anchors(cfg.BEV.ANCHORS)
lidar_dir           = os.path.join(cfg.PIXOR.DATASETS_DIR, "lidar_files/")


for fi in bev_pred_files:
	bev = np.zeros([cfg.BEV.INPUT_X, cfg.BEV.INPUT_Y, 3], dtype=np.float32)
	bev_pred = np.load(fi)[0]
	vis_tools.imshow_image(math.sigmoid(bev_pred[..., 0]))
	bev_pred_cls = math.sigmoid(bev_pred[..., 1:cfg.PIXOR.CLASSES_NUM+1])
	bev_bboxes = postprocess.parse_bev_predmap(bev_pred, bev_anchors)
	bev_bboxes = postprocess.bev_nms(bev_bboxes, cfg.BEV.DISTANCE_THRESHOLDS)
	vis_tools.imshow_bev_bbox(bev, np.array(bev_bboxes))


