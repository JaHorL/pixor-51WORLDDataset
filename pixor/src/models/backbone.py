# -*- coding: utf-8 -*-  
import models.basic_layers as bl
import tensorflow as tf


def resnet_backbone(bev_input, trainable):

    with tf.variable_scope('conv_block_1') as scope: # /1x
        bev_block = bl.convolutional(bev_input, (3, 32), trainable, 'bev_conv1')
        bev_block = bl.convolutional(bev_block, (3, 32), trainable, 'bev_conv2')
        bev_block = bl.convolutional(bev_block, (3, 64), trainable, 'bev_conv3', downsample=True)

    with tf.variable_scope('conv_block_2') as scope: # /2x
        bev_block = bl.resnet_block(bev_block, 24, 64, trainable, 'bev_res1')
        bev_block_4d = bl.convolutional(bev_block, (3, 128), trainable, 'bev_conv1', downsample=True)

    with tf.variable_scope('conv_block_3') as scope: # /4x
        bev_block =  bl.resnet_block(bev_block_4d, 48, 128, trainable, 'bev_res1')
        bev_block =  bl.resnet_block(bev_block, 48, 128, trainable, 'bev_res2')
        bev_block_8d = bl.convolutional(bev_block, (3, 192), trainable, 'bev_conv1', downsample=True)

    with tf.variable_scope('conv_block_4') as scope: # /8x
        bev_block =  bl.resnet_block(bev_block_8d, 64, 192, trainable, 'bev_res1')
        bev_block =  bl.resnet_block(bev_block, 64, 192, trainable, 'bev_res2')
        bev_block = bl.convolutional(bev_block, (3, 256), trainable, 'bev_conv1', downsample=True)

    with tf.variable_scope('conv_block_5') as scope: # /16x
        bev_block =  bl.resnet_block(bev_block, 96, 256, trainable, 'bev_res1')
        bev_block =  bl.resnet_block(bev_block, 96, 256, trainable, 'bev_res2')
        bev_block =  bl.resnet_block(bev_block, 96, 256, trainable, 'bev_res3')
        bev_block_16d =  bl.resnet_block(bev_block, 96, 256, trainable, 'bev_res4')

    with tf.variable_scope('upsample_block1') as scope: # /8x
        bev_block_16d = bl.convolutional(bev_block_16d, (1, 128), trainable, 'bev_conv1')        
        up_block_8d = bl.upsample(bev_block_16d, "deconv2d_1")
        bev_block_8d = bl.convolutional(bev_block_8d, (1, 128), trainable, 'bev_conv2')
        up_block_8d = tf.concat([up_block_8d, bev_block_8d], -1)

    with tf.variable_scope('upsample_block2') as scope: # /4x
        up_block_8d = bl.convolutional(up_block_8d, (1, 96), trainable, 'bev_conv1')
        up_block_4d = bl.upsample(up_block_8d, "deconv2d_1")
        bev_block_4d = bl.convolutional(bev_block_4d, (1, 96), trainable, 'bev_conv2')
        up_block_4d = tf.concat([up_block_4d, bev_block_4d], -1)

    return up_block_4d