import os
import random
import os.path as osp


training_list = ['LiDAR-200608-094935', 'LiDAR-200608-095604', 'LiDAR-200608-100309', 'LiDAR-200608-101211',
				 'LiDAR-200608-102318', 'LiDAR-200608-103505']
testing_list = ['LiDAR-200608-104549']
val_list = ['val']
root_dir = "/home/jhli/Data/simone/"


def gen_dataset_idx(data_list, data_type):
	total_lines = []
	for di in data_list:
		dataset_dir = root_dir + di + "/"
		for fi in os.listdir(dataset_dir):
			if ".pcd" not in fi:
				continue
			lidar_fi = osp.join(dataset_dir + fi)
			label_fi = osp.join(dataset_dir + fi[:-4] + ".json")
			calib_fi = osp.join(dataset_dir + fi[:-4] + ".txt")
			image_fi = osp.join(dataset_dir + fi[:-4] + ".png")
			if osp.exists(lidar_fi) and osp.exists(label_fi):
				total_lines.append(lidar_fi + " " + image_fi + " " + label_fi + " " + calib_fi+"\n")
			else:
				print(data_type, fi)
	save_fi = open(osp.join(root_dir, data_type+".txt"), "w")
	print(save_fi)
	random.shuffle(total_lines)
	for line in total_lines:
		save_fi.write(line)
	save_fi.close()


if __name__ == "__main__":
	gen_dataset_idx(training_list, "training")
	gen_dataset_idx(testing_list, "testing")
	gen_dataset_idx(val_list, "val")



