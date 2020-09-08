# -*- coding: utf-8 -*-  
import models.basic_layers as bl
import tensorflow as tf


def res_headnet(bev_block, cls_num, bev_bbox_dim, trainable):
    with tf.variable_scope('rh_block') as scope:
        bev_block = bl.convolutional(bev_block, (1, 96), trainable, 'bev_conv1')
        bev_block = bl.convolutional(bev_block, (3, 96), trainable, 'bev_conv2')
        bev_block = bl.convolutional(bev_block, (3, 96), trainable, 'bev_conv3')
        bev_block = bl.convolutional(bev_block, (3, 96), trainable, 'bev_conv4')
        bev_block = bl.convolutional(bev_block, (3, 96), trainable, 'bev_conv5')
        bev_obj_cls = bl.convolutional(bev_block, (1, 1 + cls_num), trainable, 'bev_obj_cls')
        bev_bbox = bl.convolutional(bev_block, (3, bev_bbox_dim), trainable, 'bev_bbox')
        bev_pred = tf.concat([bev_obj_cls, bev_bbox], -1)
    return bev_pred