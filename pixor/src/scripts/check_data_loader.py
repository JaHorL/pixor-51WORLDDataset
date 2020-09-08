import sys
sys.path.append("../")
import os
import cv2
import shutil
from data import dataset
from utils import utils
from utils import vis_tools
from tqdm import tqdm
from config.config import cfg
from data import postprocess
import numpy as np
from utils import math

if __name__ == "__main__":
    trainset = dataset.Dataset('train')
    count = 0
    for i in range(10):
        for j in range(len(trainset)):
            data = trainset.load()
            # print(len(data), len(data[0]), data[1][0].shape)
            # vis_tools.imshow_image(data[1][0][..., 0])
            count += 1