import os
import cv2
import time
import ctypes
import threading
import numpy as np
from config.config import cfg
from utils import utils
from utils import vis_tools
from utils import timer
from utils import transform
from data import labels
from data import preprocess
from loader import simone_loader
from loader.loader_config import loader_cfg

class Dataset(object):
    def __init__(self, process_type):
        if process_type == 'train':
            self.anno_path     = cfg.PIXOR.TRAIN_DATA
            self.batch_size    = cfg.TRAIN.BATCH_SIZE
            self.is_training   = True
            self.data_type     = loader_cfg.TRAINING_LOADER_FLAGS

        if process_type == 'test':
            self.anno_path     = cfg.PIXOR.TEST_DATA
            self.batch_size    = cfg.EVAL.BATCH_SIZE
            self.is_training   = True
            self.data_type     = loader_cfg.TRAINING_LOADER_FLAGS

        self.dataset_loader    = simone_loader.SimoneDatasetLoader(loader_cfg.DATASET_DIR, loader_cfg.TRAINING_LOADER_FLAGS, True)
        self.num_samples       = self.dataset_loader.get_total_num()
        # self.num_samples       = 100
        self.num_batchs        = int(np.ceil(self.num_samples / self.batch_size)-2)
        self.batch_count       = 0
        self.is_use_thread     = cfg.PIXOR.IS_USE_THREAD
        self.bev_anchors       = self.load_anchors(cfg.BEV.ANCHORS)

        self.loader_need_exit = 0
        self.timer = timer.Timer()
        self.per_step_ano = []
        if self.is_use_thread:
            self.prepr_data = []
            self.max_cache_size = 10
            self.lodaer_processing = threading.Thread(target=self.loader)
            self.lodaer_processing.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.loader_need_exit = True
        print('set loader_need_exit True')
        self.lodaer_processing.join()
        print('exit lodaer_processing')

    def __len__(self):
        return self.num_batchs

    def load_anchors(self, anchors_path):
        with open(anchors_path) as f:
            anchors = f.readlines()
        new_anchors = np.zeros([len(anchors), len(anchors[0].split())], dtype=np.float32)
        for i in range(len(anchors)):
            new_anchors[i] = np.array(anchors[i].split(), dtype=np.float32)
        return new_anchors

    def preprocess_data(self):
        batch_bev = np.zeros((self.batch_size, cfg.BEV.INPUT_X, cfg.BEV.INPUT_Y, cfg.BEV.INPUT_Z), dtype=np.float32)
        batch_bev_label = np.zeros((self.batch_size, cfg.BEV.OUTPUT_X, cfg.BEV.OUTPUT_Y, cfg.BEV.LABEL_Z), 
                                    dtype=np.float32)
        for i in range(self.batch_size):
            data_info = self.dataset_loader.next()
            batch_bev_label[i, ...] = labels.create_bev_label(data_info['pcd_annos'], self.bev_anchors).astype(np.float32)
            batch_bev[i, ...] = preprocess.lidar_preprocess(data_info['pointcloud'])
        self.batch_count += 1
        return [batch_bev, batch_bev_label]

    def loader(self):
        while(not self.loader_need_exit):
            if len(self.prepr_data) < self.max_cache_size: 
                self.prepr_data.append(self.preprocess_data())
            else:
                time.sleep(0.01)
                self.loader_need_exit = False

    def load(self):
        if self.is_use_thread:
            while len(self.prepr_data) == 0:
                time.sleep(0.01)
            data_ori = self.prepr_data.pop()
        else:
            data_ori = self.preprocess_data()
        if self.batch_count >= self.num_batchs:
            self.batch_count = 0
        return data_ori
                                       
                                                   

if __name__ == "__main__":
	pass