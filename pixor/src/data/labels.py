import cv2
import numpy as np
import math
from utils import transform
from config.config import cfg
from utils import vis_tools




def create_bev_label(annos, anchors):
    objectness_class_map = np.zeros((cfg.BEV.OUTPUT_X, cfg.BEV.OUTPUT_Y, 3), dtype=np.float32)
    bev_center_map = np.zeros((cfg.BEV.OUTPUT_X, cfg.BEV.OUTPUT_Y, 3), dtype=np.float32)
    for i in range(len(annos)):
        anno = annos[i]
        anno['relativeRot'][2] = -(anno["relativeRot"][2]+np.pi/2)
        rz = transform.ry_to_rz(anno['relativeRot'][2])
        location = anno['relativePos']
        anno['size'].reverse()
        bev_location = transform.location_lidar2bevlabel(location)
        xx, yy, _ = bev_location
        hwl = np.array(anno['size']) / (cfg.BEV.X_RESOLUTION * 4)
        box = transform.bevbox_compose(xx, yy, hwl[1], hwl[2], rz)
        cls_type = cfg.PIXOR.KITTI_CLS_TYPE[anno['type']]
        cv2.fillConvexPoly(objectness_class_map, box, [i+1, cfg.PIXOR.KITTI_CLASSES_LIST.index(cls_type), 0.0])
        cv2.fillConvexPoly(bev_center_map, box, [float(xx), float(yy), 0.0])
    # vis_tools.imshow_image(objectness_class_map)
    bev_label = np.zeros([cfg.BEV.OUTPUT_X, cfg.BEV.OUTPUT_Y, cfg.BEV.LABEL_Z], np.float32)
    for i in range(cfg.BEV.OUTPUT_X):
        for j in range(cfg.BEV.OUTPUT_Y):
            if objectness_class_map[i][j][0] < 0.1: continue
            type_id = int(objectness_class_map[i][j][1])
            idx = int(objectness_class_map[i][j][0]-1)
            anno = annos[idx]
            rz = transform.ry_to_rz(anno['relativeRot'][2])
            dim = anno['size']
            theta = rz if rz < 0 else rz + 3.14
            center_x, center_y, _ = bev_center_map[i][j]
            delta_x = (i-center_x) / anchors[type_id][3]
            delta_y = (j-center_y) / anchors[type_id][4]
            offset_xy = np.sqrt(pow((center_x-i), 2) + pow((center_y-j), 2))
            prob = pow(cfg.BEV.PROB_DECAY, offset_xy)
            h, w, l= dim / anchors[type_id][:3]
            box = np.array([delta_x, delta_y, h, w, l, theta], np.float32)
            bev_label[i][j][:3] = np.array([1, type_id, prob])
            bev_label[i][j][3+type_id*cfg.BEV.BBOX_DIM:3+(type_id+1)*cfg.BEV.BBOX_DIM] = box
    return bev_label




