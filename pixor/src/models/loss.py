import tensorflow as tf 
from config.config import cfg


def smooth_l1(deltas, targets, sigma=2.0):
	'''
	ResultLoss = outside_weights * SmoothL1(inside_weights * (box_pred - box_targets))
	SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
					|x| - 0.5 / sigma^2,    otherwise
	'''
	sigma2 = sigma * sigma
	diffs  =  tf.subtract(deltas, targets)
	l1_signs = tf.cast(tf.less(tf.abs(diffs), 1.0 / sigma2), tf.float32)

	l1_option1 = tf.multiply(diffs, diffs) * 0.5 * sigma2
	l1_option2 = tf.abs(diffs) - 0.5 / sigma2
	l1_add = tf.multiply(l1_option1, l1_signs) + \
						tf.multiply(l1_option2, 1-l1_signs)
	l1 = l1_add

	return l1


def bev_loss(pred, label, bev_loss_scale):
	cls_num = cfg.PIXOR.CLASSES_NUM
	epsilon = cfg.PIXOR.EPSILON
	shape = [-1, cfg.BEV.OUTPUT_X, cfg.BEV.OUTPUT_Y, cfg.BEV.OUTPUT_Z]
	pred = tf.reshape(pred, shape)
	mask = tf.cast(label[...,0], tf.bool)
	# raise Exception(mask)
	with tf.name_scope('mask'):
		masked_label = tf.boolean_mask(label, mask)
		masked_pred  = tf.boolean_mask(pred, mask)
		masked_neg_pred = tf.boolean_mask(pred, tf.logical_not(mask))
		# raise Exception(pred, label, masked_label, masked_pred, masked_neg_pred)

	with tf.name_scope('pred'):
		pred_o = tf.sigmoid(masked_pred[...,0])
		pred_no_o = tf.sigmoid(masked_neg_pred[...,0])
		pred_c = tf.sigmoid(masked_pred[..., 1:cls_num+1])
		box_shape = (-1, cls_num, cfg.BEV.BBOX_DIM)
		pred_box = tf.reshape(masked_pred[..., 1+cls_num:], shape = box_shape)
		pred_xyhwl = pred_box[...,:-1]
		pred_re = tf.cos(pred_box[...,-1])
		pred_im = tf.sin(pred_box[...,-1])

	with tf.name_scope('label'):
		label_c = tf.one_hot(tf.cast(masked_label[..., 1], tf.int32), depth=cls_num)
		label_c_scale = tf.gather(bev_loss_scale, tf.cast(masked_label[..., 1], tf.int32))
		label_prob = masked_label[..., 2]
		box_shape = (-1, cls_num, cfg.BEV.BBOX_DIM)
		label_box = tf.reshape(masked_label[..., 3:], shape = box_shape)
		label_xyhwl = label_box[...,:-1]
		label_re = tf.cos(label_box[...,-1])
		label_im = tf.sin(label_box[...,-1])

	with tf.name_scope('loss'):
		xywhl_loss = tf.reduce_sum(tf.reduce_sum(smooth_l1(pred_xyhwl, label_xyhwl), axis=-1)*label_c)
		re_loss = tf.reduce_sum(smooth_l1(pred_re, label_re)*label_c) * 2
		im_loss = tf.reduce_sum(smooth_l1(pred_im, label_im)*label_c) * 2
		has_obj_loss = tf.reduce_sum(-tf.log(pred_o + epsilon) * label_prob) * 5
		no_obj_loss = tf.reduce_sum(-tf.log(1 - pred_no_o + epsilon)) * 0.1
		cls_loss = tf.reduce_sum(tf.reduce_sum(-tf.log(pred_c + epsilon)*label_c, axis=-1)*label_c_scale) * 5 + \
					tf.reduce_sum(-tf.log(1-pred_c + epsilon)*(1-label_c))
		cls_loss = cls_loss / cfg.TRAIN.BATCH_SIZE
		bbox_loss = (xywhl_loss + re_loss + im_loss) / cfg.TRAIN.BATCH_SIZE
		objness_loss = (has_obj_loss + no_obj_loss) / cfg.TRAIN.BATCH_SIZE
		total_loss = bbox_loss * 5 + objness_loss + cls_loss
		
	return total_loss, objness_loss, cls_loss, bbox_loss

