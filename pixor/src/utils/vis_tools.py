import cv2
import os
import numpy as np
from config.config import cfg
from utils import transform	

def draw_img_bboxes2d(img, bboxes, cls_types):
	bboxes = bboxes.astype(np.int32)
	num=len(bboxes)
	for n in range(num):
		b = bboxes[n]
		color = cfg.PIXOR.CLASSES_COLOR[int(cls_types[n])]
		cv2.line(img, (b[0],b[1]), (b[2],b[1]), color, 2, cv2.LINE_AA)
		cv2.line(img, (b[0],b[1]), (b[0],b[3]), color, 2, cv2.LINE_AA)
		cv2.line(img, (b[0],b[3]), (b[2],b[3]), color, 2, cv2.LINE_AA)
		cv2.line(img, (b[2],b[1]), (b[2],b[3]), color, 2, cv2.LINE_AA)
	return  img


def draw_img_bboxes3d(img, bboxes, cls_types, color=(255,0,255)):
	num=len(bboxes)
	for n in range(num):
		b = bboxes[n]
		color = cfg.PIXOR.CLASSES_COLOR[int(cls_types[n])]
		for k in range(0,4):
			i,j=k,(k+1)%4
			cv2.line(img, (b[i,0],b[i,1]), (b[j,0],b[j,1]), color, 2, cv2.LINE_AA)
			i,j=k+4,(k+1)%4 + 4
			cv2.line(img, (b[i,0],b[i,1]), (b[j,0],b[j,1]), color, 2, cv2.LINE_AA)
			i,j=k,k+4
			cv2.line(img, (b[i,0],b[i,1]), (b[j,0],b[j,1]), color, 2, cv2.LINE_AA)
	return img


def draw_bev_bboxes(bev, bev_bboxes, cls_types, color=(255,0,0)):
	bev = bev.copy()
	num=len(bev_bboxes)
	for n in range(num):
		b = bev_bboxes[n]
		color = cfg.PIXOR.CLASSES_COLOR[int(cls_types[n])]
		cv2.line(bev, (b[0,0],b[0,1]), (b[1,0],b[1,1]), color, 2, cv2.LINE_AA)
		cv2.line(bev, (b[1,0],b[1,1]), (b[2,0],b[2,1]), color, 2, cv2.LINE_AA)
		cv2.line(bev, (b[2,0],b[2,1]), (b[3,0],b[3,1]), color, 2, cv2.LINE_AA)
		cv2.line(bev, (b[3,0],b[3,1]), (b[0,0],b[0,1]), color, 2, cv2.LINE_AA)
	return  bev
	

def imshow_image(img, new_size=None, name=None):
		if not name:
				name = 'img1'
		if new_size:
				img = cv2.resize(img, new_size)
		cv2.imshow(name, img)
		cv2.moveWindow(name, 0, 0)
		cv2.waitKey(0)
		cv2.destroyAllWindows()


def write_image(img, save_path, idx, size=None):
		name = save_path + "/" + idx + '.jpg'
		if size:
				img = img.resize(size)
		# img = img.astype(np.int32)
		cv2.imwrite(name, img)

def imshow_img_bbox(img, bboxes):
    bboxes = np.array(bboxes)
    if(len(bboxes) == 0): 
        return
    corners = np.stack([bboxes[..., 2], bboxes[..., 3], bboxes[..., 4], bboxes[..., 5]], axis=-1)
    img = draw_img_bboxes2d(img, corners, bboxes[..., 0])
    imshow_image(img)


def imshow_bev_bbox(bev, bboxes):
    bboxes = np.array(bboxes)
    if(len(bboxes) == 0):
        return
    bev_bboxes = []
    for b in bboxes:
        xyz = transform.location_lidar2bev(b[2:5])
        w = b[5] / cfg.BEV.X_RESOLUTION
        l = b[6] / cfg.BEV.Y_RESOLUTION
        bev_bbox = transform.bevbox_compose(xyz[0], xyz[1], w, l, b[7])
        bev_bboxes.append(bev_bbox)
    bev = draw_bev_bboxes(bev, bev_bboxes, bboxes[..., 0])
    imshow_image(bev)

        
def imwrite_img_bbox(img, bboxes, save_path, idx):
    bboxes = np.array(bboxes)
    if(len(bboxes) == 0): 
        return
    corners = np.stack([bboxes[..., 2], bboxes[..., 3], bboxes[..., 4], bboxes[..., 5]], axis=-1)
    img = draw_img_bboxes2d(img, corners, bboxes[..., 0])
    write_image(img, save_path, idx)


def imwrite_bev_bbox(bev, bboxes, save_path, idx):
    bboxes = np.array(bboxes)
    if(len(bboxes) == 0):
        return
    bev_bboxes = []
    for b in bboxes:
        xyz = transform.location_lidar2bev(b[2:5])
        w = b[5] / cfg.BEV.X_RESOLUTION
        l = b[6] / cfg.BEV.Y_RESOLUTION
        bev_bbox = transform.bevbox_compose(xyz[0], xyz[1], w, l, b[7])
        bev_bboxes.append(bev_bbox)
    bev = draw_bev_bboxes(bev, bev_bboxes, bboxes[..., 0])
    write_image(bev, save_path, idx)
