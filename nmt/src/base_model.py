import os

import tensorflow as tf


class BaseModel(object):
    def __init__(self, config):
        self.config = config
        self.sess = None
        self.saver = None
        self.logger = config.logger

    def init_session(self):
        self.logger.info('Initialise TF Session')
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def save_session(self, step=None):
        self.logger.info()
        if not os.path.exists(self.config.model_dir):
            os.mkdir(self.config.model_dir)
        self.saver.save(sess=self.sess, save_path=self.config.model_dir, global_step=step)

    def restore_session(self):
        self.logger.info()
        self.saver = tf.train.Saver()
        self.saver.restore(sess=self.sess, save_path=self.config.model_dir)

    def add_summary(self):
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.config.summary_dir, self.sess.graph)

    def add_training_op(self, optimizer_param, lr, loss, clip=-1):
        """
        :param optimizer_param: param for choosing method of optimizer
        :param lr: learning rate
        :param loss: loss
        :param clip: clip
        :return:
        """
        _optimizer_param = optimizer_param.lower()
        if _optimizer_param == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(lr)
        else:
            raise NotImplementedError('Optimizer should choose in [sgd]')

        if clip > 0 :
            gradients, variables = zip(*optimizer.compute_gradients(loss=loss))
