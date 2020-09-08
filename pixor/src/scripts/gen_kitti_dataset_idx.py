import os
import random
import os.path as osp

dataset_dir = "/home/jhli/Data/kitti"
print(dataset_dir)
lidar_dir = osp.join(dataset_dir, "lidar_files")
calib_dir = osp.join(dataset_dir, "calib_files")
image_dir = osp.join(dataset_dir, "image_files")
testing_dir = osp.join(dataset_dir, "label_files/testing")
training_dir = osp.join(dataset_dir, "label_files/training")
val_dir = osp.join(dataset_dir, "label_files/val")


def gen_dataset_idx(dir_name, data_type):
  total_lines = []
  for fi in os.listdir(dir_name):
    label_fi = osp.join(dir_name, fi)
    if ".txt" not in label_fi: 
      print(label_fi)
      continue
    lidar_fi = osp.join(lidar_dir, fi[:-4]+".bin")
    calib_fi = osp.join(calib_dir, fi[:-4]+".txt")
    image_fi = osp.join(image_dir, fi[:-4]+".png")
    if osp.exists(lidar_fi) and osp.exists(calib_dir) and osp.exists(image_dir):
      total_lines.append(lidar_fi + " " + image_fi + " " + label_fi + " " + calib_fi+"\n")
    else:
      print(data_type, fi)
  save_fi = open(osp.join(dataset_dir, data_type+".txt"), "w")
  print(save_fi)
  random.shuffle(total_lines)
  for line in total_lines:
    save_fi.write(line)
  save_fi.close()


if __name__ == "__main__":
  gen_dataset_idx(training_dir, "training")
  gen_dataset_idx(testing_dir, "testing")
  gen_dataset_idx(val_dir, "val")
