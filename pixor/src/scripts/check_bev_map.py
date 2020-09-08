import sys
sys.path.append("../")
import numpy as np
import cv2
from data import dataset
from data import preprocess
from utils import vis_tools
from tqdm import tqdm
from data import loader
from config.config import cfg


def merge_obj_bboxes(bboxes, cls_type):
	new_box = np.zeros(bboxes.shape[-1])
	new_box[0] = cls_type
	# print(bboxes[..., -1])
	new_box[1] = np.mean(bboxes[..., 1])
	new_box[2:] = np.sum(bboxes[..., 2:]*bboxes[..., 1][..., np.newaxis], axis=0)
	sum_of_prob = np.sum(bboxes[..., 1])
	new_box[2:] = new_box[2:] / sum_of_prob
	area = new_box[5] * new_box[6]        
	if sum_of_prob / area < 1:
		return []
	return new_box


def is_close_center(bbox1, bbox2, threshold):
	xy = bbox2 - bbox1
	distance = np.sqrt(xy[:,0]*xy[:,0]+ xy[:,1]*xy[:,1])
	return distance < threshold


def cal_tpfnfp(gts, results):
	tp = []
	fp = []
	fn = []
	for bbox in results:
		thres = max(thresholds[int(bbox[])])
		distance_mask = is_close_center(bbox[..., 2:4], gts[..., 2:4])
		obj_bbox = gts[distance_mask]
		if obj_bbox:
			tp.append(obj_bbox)
		else:
			fp.append(bbox)
		gts = gts[np.logical_not(distance_mask)]

	fn = gts
	return np.array(tp), np.array(fp), np.array(fn)
		


if __name__ == "__main__":
	lidar_preprocessor  = preprocess.LidarPreprocessor()
	trainset            = dataset.Dataset(lidar_preprocessor, 'train')
	pbar                = tqdm(trainset)
	for data, anno in pbar:
		frame_id, calib, img, pc, gts = loader.load_input_data(anno, 'simone')
		types, dimensions, box2d_corners, locations, rzs, ids, _ = [np.array(gt) for gt in gts]
		gts = np.concatenate([ids[:, np.newaxis], types[:, np.newaxis], locations, dimensions, rzs[:, np.newaxis]], axis=-1)
		tp, fp, fn = cal_tpfnfp(gts, gts)
		print(tp, fp, fn)