import os
import io
import time
import shutil
import cv2
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from config.config import cfg
from utils import utils
from data import dataset
from data import preprocess
from data import postprocess
from utils import vis_tools
from models import pixor_network
from utils import math
import json
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession




class predicter(object):

    def __init__(self):
        self.initial_weight      = cfg.EVAL.WEIGHT
        self.time                = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        self.moving_ave_decay    = cfg.PIXOR.MOVING_AVE_DECAY
        self.eval_logdir        = "./data/logs/eval"
        self.lidar_preprocessor  = preprocess.LidarPreprocessor()
        self.evalset             = dataset.Dataset(self.lidar_preprocessor, 'test')
        self.output_dir          = cfg.EVAL.OUTPUT_PRED_PATH
        self.bev_anchors         = loader.load_anchors(cfg.BEV.ANCHORS)

        with tf.name_scope('model'):
            self.model               = pixor_network.PixorNetwork()
            self.net                 = self.model.load()
            self.bev_pred            = self.net['bev_pred']

        config = ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = InteractiveSession(config=config)
        self.saver = tf.train.Saver()#ema_obj.variables_to_restore())
        self.saver.restore(self.sess, self.initial_weight)


    def predict(self):
        bev_imwrite_path = os.path.join(self.output_dir, "bev_imshow_result/")
        bev_result_path  = os.path.join(self.output_dir, "bev_result/")
        if os.path.exists(bev_imwrite_path):
            shutil.rmtree(bev_imwrite_path)
        os.mkdir(bev_imwrite_path)
        if os.path.exists(bev_result_path):
            shutil.rmtree(bev_result_path)
        os.mkdir(bev_result_path)
        for step in range(len(self.evalset)):
            print(step, "/", len(self.evalset))
            eval_result = self.evalset.load()
            bev_pred = self.sess.run(self.bev_pred,    
                                     feed_dict={self.net["bev_input"]: eval_result[0],
                                     self.net["trainable"]: False})[0][0]

            bev_bboxes = postprocess.parse_bev_predmap(bev_pred, self.bev_anchors)
            bev_bboxes = postprocess.bev_nms(bev_bboxes, cfg.BEV.DISTANCE_THRESHOLDS)


if __name__ == "__main__":
    predicter = predicter()
    predicter.predict()