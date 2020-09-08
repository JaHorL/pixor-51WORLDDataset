import os
import io
import time
import shutil
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from utils import timer
from config.config import cfg
from data import dataset
from data import preprocess
from models import pixor_network
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession






class Trainer(object):

    def __init__(self):
        self.learn_rate_init     = cfg.TRAIN.LEARN_RATE_INIT
        self.learn_rate_end      = cfg.TRAIN.LEARN_RATE_END
        self.first_stage_epochs  = cfg.TRAIN.FRIST_STAGE_EPOCHS
        self.second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS
        self.warmup_periods      = cfg.TRAIN.WARMUP_EPOCHS
        self.initial_weight      = cfg.TRAIN.PRETRAIN_WEIGHT
        self.time                = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        self.moving_ave_decay    = cfg.PIXOR.MOVING_AVE_DECAY
        self.train_logdir        = "./data/log/train"
        self.trainset            = dataset.Dataset('train')
        self.valset              = dataset.Dataset('test')
        self.steps_per_period    = len(self.trainset)
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = InteractiveSession(config=config)
        self.timer = timer.Timer()
        # self.sess                = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

        with tf.name_scope('model'):
            self.model               = pixor_network.PixorNetwork()
            self.net                 = self.model.load()
            self.net_var             = tf.global_variables()
            self.loss                = self.net["pixor_loss"]

        with tf.name_scope('learn_rate'):
            self.global_step = tf.Variable(1.0, dtype=tf.float64, trainable=False, name='global_step')
            warmup_steps = tf.constant(self.warmup_periods * self.steps_per_period,
                                        dtype=tf.float64, name='warmup_steps')
            train_steps = tf.constant( (self.first_stage_epochs + self.second_stage_epochs)* self.steps_per_period,
                                        dtype=tf.float64, name='train_steps')
            self.learn_rate = tf.cond(
                pred=self.global_step < warmup_steps,
                true_fn=lambda: self.global_step / warmup_steps * self.learn_rate_init,
                false_fn=lambda: self.learn_rate_end + 0.5 * (self.learn_rate_init - self.learn_rate_end) *
                         (1 + tf.cos((self.global_step - warmup_steps) / (train_steps - warmup_steps) * np.pi))
                         )
            global_step_update = tf.assign_add(self.global_step, 1.0)

        with tf.name_scope("define_weight_decay"):
            moving_ave = tf.train.ExponentialMovingAverage(self.moving_ave_decay).apply(tf.trainable_variables())

        with tf.name_scope("define_first_stage_train"):
            self.first_stage_trainable_var_list = []
            for var in tf.trainable_variables():
                var_name = var.op.name
                var_name_mess = str(var_name).split('/')
                if var_name_mess[0] in ["pixor_headnet"]:
                    self.first_stage_trainable_var_list.append(var)
            first_stage_optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss,
                                                        var_list=self.first_stage_trainable_var_list)
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([first_stage_optimizer, global_step_update]):
                    with tf.control_dependencies([moving_ave]):
                        self.train_op_with_frozen_variables = tf.no_op()

        with tf.name_scope("define_second_stage_train"):
            second_stage_trainable_var_list = tf.trainable_variables()
            second_stage_optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss,
                                                        var_list=second_stage_trainable_var_list)
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([second_stage_optimizer, global_step_update]):
                    with tf.control_dependencies([moving_ave]):
                        self.train_op_with_all_variables = tf.no_op()

        with tf.name_scope('loader_and_saver'):
            self.loader = tf.train.Saver(self.net_var)
            self.saver  = tf.train.Saver(tf.global_variables(), max_to_keep=10)

        with tf.name_scope('summary'):
            tf.summary.scalar("learn_rate",    self.learn_rate)
            tf.summary.scalar("pixor_loss",    self.net["pixor_loss"])
            tf.summary.scalar("bev_obj_loss",  self.net["bev_obj_loss"])
            tf.summary.scalar("bev_cls_loss",  self.net["bev_cls_loss"])
            tf.summary.scalar("bev_bbox_loss", self.net["bev_bbox_loss"])
            logdir = "../logs/tensorboard"
            if os.path.exists(logdir): 
                shutil.rmtree(logdir)
            os.mkdir(logdir)
            self.write_op = tf.summary.merge_all()
            self.summary_writer  = tf.summary.FileWriter(logdir, graph=self.sess.graph)
        bev_pred_dir = cfg.PIXOR.LOG_DIR+"/pred/bev_pred/"
        if os.path.exists(bev_pred_dir): 
            shutil.rmtree(bev_pred_dir)
        os.mkdir(bev_pred_dir)


    def train(self):
        self.sess.run(tf.global_variables_initializer())
        try:
            print('=> Restoring weights from: %s ... ' % self.initial_weight)
            self.loader.restore(self.sess, self.initial_weight)
        except:
            print('=> %s does not exist !!!' % self.initial_weight)
            print('=> Now it starts to train pixor from scratch ...')
            self.first_stage_epochs = 0

        for epoch in range(1, 1+self.first_stage_epochs+self.second_stage_epochs):
            if epoch <= self.first_stage_epochs:
                train_op = self.train_op_with_frozen_variables
            else:
                train_op = self.train_op_with_all_variables

            train_epoch_loss = []
            last_log_time = time.time()
            print("======================================> Epoch:%d <======================================"%epoch)
            for _ in range(self.steps_per_period):
                train_data = self.trainset.load()
                _, summary, train_step_loss, global_step_val = self.sess.run(
                    [train_op, self.write_op, self.loss, self.global_step],feed_dict={
                                                self.net["bev_input"]:   train_data[0],
                                                self.net["bev_label"]:   train_data[1],
                                                self.net["bev_loss_scale"]: cfg.BEV.LOSS_SCALE,
                                                self.net["trainable"]:   True
                    })
                train_epoch_loss.append(train_step_loss)
                self.summary_writer.add_summary(summary, global_step_val)
                if global_step_val % cfg.PIXOR.PRINTING_STEPS == 0:
                    log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                    print("======> Epoch:%2d Time:%s step: %d/%d During Time: %.2f Train loss: %.2f" 
                         %(epoch, log_time, global_step_val%self.steps_per_period, self.steps_per_period, 
                         time.time()-last_log_time, np.mean(train_epoch_loss[-cfg.PIXOR.PRINTING_STEPS:])))
                    last_log_time = time.time()
                if global_step_val % cfg.TRAIN.SAVING_STEPS==0:
                    val_epoch_loss = []
                    print("valing...")
                    for _ in range(len(self.valset)):
                        val_data = self.valset.load()
                        val_step_loss, bev_pred = self.sess.run(
                                [self.loss, self.net['bev_pred']],feed_dict={
                                                self.net["bev_input"]:   val_data[0],
                                                self.net["bev_label"]:   val_data[1],
                                                self.net["bev_loss_scale"]: cfg.BEV.LOSS_SCALE,
                                                self.net["trainable"]:   True
                        })
                        val_epoch_loss.append(val_step_loss)
                    print("saving...")
                    train_epoch_loss_m, val_epoch_loss_m = np.mean(train_epoch_loss), np.mean(val_epoch_loss)
                    ckpt_file = "../checkpoint/pixor_val_loss=%.4f.ckpt" % val_epoch_loss_m
                    log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                    print("======> Epoch:%2d Time: %s Train loss: %.2f val loss: %.2f Saving %s ..."
                                    %(epoch, log_time, train_epoch_loss_m, val_epoch_loss_m, ckpt_file))
                    self.saver.save(self.sess, ckpt_file, global_step=epoch)
        print("saving...")
        save_time = time.asctime(time.localtime(time.time()))
        ckpt_file = "../checkpoint/pixor_last_epoch-%s.ckpt" % save_time
        self.saver.save(self.sess, ckpt_file, global_step=epoch)



if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()