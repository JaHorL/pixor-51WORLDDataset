import os.path as osp
import numpy as np
from easydict import EasyDict as edict


__C                              = edict()
cfg                              = __C


__C.PIXOR                        = edict()
__C.PIXOR.DATASET_TYPE           = "simone" #"simone"
__C.PIXOR.KITTI_CLASSES_LIST     = ['Car','Van','Truck','Pedestrian','Cyclist','Misc']
__C.PIXOR.SIMONE_CLASSES_LIST    = ['6','19','18','4','17','-1'] #'Car','Bus','Truck','Pedestrian','Cyclist','-1'
__C.PIXOR.CLASSES_COLOR          = [(255,0,0),(255,255,0),(255,0,255),(0,255,0),(128,64,255),(0,255,255)]
__C.PIXOR.CLASSES_NUM            = len(__C.PIXOR.SIMONE_CLASSES_LIST)
__C.PIXOR.EPSILON                = 0.00001
__C.PIXOR.ROOT_DIR               = osp.abspath(osp.join(osp.dirname(__file__), '../..'))
__C.PIXOR.LOG_DIR                = osp.join(__C.PIXOR.ROOT_DIR, 'logs')
__C.PIXOR.DATASETS_DIR           = "/home/jhli/Data/" + __C.PIXOR.DATASET_TYPE 
__C.PIXOR.TRAIN_DATA             = osp.join(__C.PIXOR.DATASETS_DIR, "training.txt")
__C.PIXOR.TEST_DATA              = osp.join(__C.PIXOR.DATASETS_DIR, "training.txt")
__C.PIXOR.MOVING_AVE_DECAY       = 0.9995
__C.PIXOR.IS_USE_THREAD          = True
__C.PIXOR.POINTS_THRESHOLDS      = [40,80,80,20,20,30]
__C.PIXOR.PRINTING_STEPS         = 20
__C.PIXOR.LIDAR_HEIGHT           = 1.8
__C.PIXOR.KITTI_CLS_TYPE         = dict( 
                                        Pedestrian = 'Pedestrian',
                                        Car = 'Car',
                                        Rider = 'Cyclist',
                                        TrafficLight = 'Misc',
                                        Truck ='Truck',
                                        Bus = 'Van',
                                        SpecialVehicle = 'Misc',
                                        SpeedLimitSign = 'Misc',
                                        RoadObstacle = 'Misc'
                                        )


__C.BEV                       = edict()
__C.BEV.ANCHORS               = __C.PIXOR.ROOT_DIR + "/src/config/anchors/bev_anchors.txt"
__C.BEV.LOSS_SCALE            = np.array([2.0, 1.0, 1.0, 5.0, 5.0, 5.0])
__C.BEV.X_MAX                 = 72
__C.BEV.X_MIN                 = 0
__C.BEV.Y_MAX                 = 36
__C.BEV.Y_MIN                 = -36
__C.BEV.Z_MAX                 = 1.2
__C.BEV.Z_MIN                 = -2.0  
__C.BEV.X_RESOLUTION          = 0.15
__C.BEV.Y_RESOLUTION          = 0.15
__C.BEV.Z_RESOLUTION          = 0.2
__C.BEV.Z_STATISTIC_DIM       = 6
__C.BEV.STRIDE                = 4
__C.BEV.BBOX_DIM              = 6
__C.BEV.PROB_DECAY            = 0.98
__C.BEV.IS_LIDAR_AUG          = False
__C.BEV.INPUT_X               = int((__C.BEV.X_MAX - __C.BEV.X_MIN) / __C.BEV.X_RESOLUTION)
__C.BEV.INPUT_Y               = int((__C.BEV.Y_MAX - __C.BEV.Y_MIN) / __C.BEV.Y_RESOLUTION)
__C.BEV.LAYERED_DIM           = int((__C.BEV.Z_MAX - __C.BEV.Z_MIN)/ __C.BEV.Z_RESOLUTION)
__C.BEV.INPUT_Z               = 3
__C.BEV.LABEL_Z               = int(1 + 1 + 1 + __C.BEV.BBOX_DIM * __C.PIXOR.CLASSES_NUM)
__C.BEV.OUTPUT_X              = int(__C.BEV.INPUT_X / __C.BEV.STRIDE)
__C.BEV.OUTPUT_Y              = int(__C.BEV.INPUT_Y / __C.BEV.STRIDE)
__C.BEV.OUTPUT_Z              = __C.PIXOR.CLASSES_NUM * cfg.BEV.BBOX_DIM + __C.PIXOR.CLASSES_NUM + 1 
__C.BEV.DISTANCE_THRESHOLDS   = [1.5, 3.0, 3.0, 0.5, 0.5, 1.5]


__C.IMAGE                     = edict()
__C.IMAGE.INPUT_H             = 192
__C.IMAGE.INPUT_W             = 640
__C.IMAGE.H_SCALE_RATIO       = __C.IMAGE.INPUT_H / 375
__C.IMAGE.W_SCALE_RATIO       = __C.IMAGE.INPUT_W / 1242


__C.TRAIN                     = edict()

__C.TRAIN.PRETRAIN_WEIGHT     = "../checkpoint/pixor_val_loss=130.9991.ckpt-25"
__C.TRAIN.BATCH_SIZE          = 1
__C.TRAIN.SAVING_STEPS        = int(4000 / __C.TRAIN.BATCH_SIZE)
# __C.TRAIN.SAVING_STEPS        = 50
__C.TRAIN.FRIST_STAGE_EPOCHS  = 0
__C.TRAIN.SECOND_STAGE_EPOCHS = 25
__C.TRAIN.WARMUP_EPOCHS       = 0
__C.TRAIN.LEARN_RATE_INIT     = 1e-3
__C.TRAIN.LEARN_RATE_END      = 1e-5
__C.TRAIN.IS_DATA_AUG         = True


__C.EVAL                      = edict()
__C.EVAL.BATCH_SIZE           = 1
__C.EVAL.WEIGHT               = "../checkpoint/pixor_val_loss=130.9991.ckpt-25"
__C.EVAL.OUTPUT_GT_PATH       = osp.join(__C.PIXOR.LOG_DIR, "gt")
__C.EVAL.OUTPUT_PRED_PATH     = osp.join(__C.PIXOR.LOG_DIR, "pred")


