# -*- coding: utf-8 -*-  
import models.basic_layers as bl
import tensorflow as tf
from config.config import cfg
from models import backbone
from models import headnet
from models import loss


class PixorNetwork(object):
    def __init__(self):
        self.cls_num = cfg.PIXOR.CLASSES_NUM
        self.bev_bbox_dim = self.cls_num * cfg.BEV.BBOX_DIM

    def net(self, bev_input, trainable):
        with tf.variable_scope('pixor_backbone') as scope:
            bev_block = backbone.resnet_backbone(bev_input, trainable)
        with tf.variable_scope('pixor_headnet') as scope:
            bev_pred = headnet.res_headnet(bev_block, self.cls_num, self.bev_bbox_dim, trainable)
        return bev_pred, 


    def load(self):
        bev_shape = [None, cfg.BEV.INPUT_X, cfg.BEV.INPUT_Y, cfg.BEV.INPUT_Z]
        bev_label_shape = [None, cfg.BEV.OUTPUT_X, cfg.BEV.OUTPUT_Y, cfg.BEV.LABEL_Z]
        bev_input = tf.placeholder(dtype=tf.float32, shape=bev_shape, name='bev_input_placeholder')
        bev_label = tf.placeholder(dtype=tf.float32, shape=bev_label_shape, name='bev_label_placeholder')
        bev_loss_scale = tf.placeholder(dtype=tf.float32, shape=[6], name='bev_loss_scale')
        trainable = tf.placeholder(dtype=tf.bool, name='training')
        bev_pred = self.net(bev_input, trainable)

        with tf.variable_scope('bev_loss') as scope:
            bev_loss = loss.bev_loss(bev_pred, bev_label, bev_loss_scale)

        return {'bev_input':bev_input,
                'bev_label':bev_label,
                'bev_pred':bev_pred,
                'bev_loss_scale':bev_loss_scale,
                'trainable':trainable,
                'pixor_loss': bev_loss[0],
                'bev_obj_loss': bev_loss[1],
                'bev_cls_loss': bev_loss[2],
                'bev_bbox_loss': bev_loss[3],
                }



