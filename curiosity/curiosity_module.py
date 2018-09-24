###############################################################################
# idea adapted from:
# Deepak Pathak, Pulkit Agrawal, Alexei A. Efros, Trevor Darrell
# University of California, Berkeley
# Curiosity-driven Exploration by Self-supervised Prediction

# added by Paula
###############################################################################

import numpy as np
import os
import tensorflow as tf

from curiosity import model_prediction_curiosity as mpc
from utils import Settings


class Curious(object):
    def __init__(self):
        tf.reset_default_graph()
        self.learning_rate = 0.001
        self.forward_loss_wt = 0.2
        self.feat_size = 200
        self.num_actions = 16
        self.num_belief_states = 268
        self.layer2 = 200

        if Settings.config.has_option("eval", "feat_size"):
            self.feat_size = Settings.config.getint("eval", "feat_size")

        with tf.variable_scope('curiosity', reuse=tf.AUTO_REUSE):
            self.predictor = mpc.StateActionPredictor(self.num_belief_states, self.num_actions,
                                                      feature_size=self.feat_size, layer2=self.layer2)

            self.predloss = self.predictor.invloss * (1 - self.forward_loss_wt) + \
                            self.predictor.forwardloss * self.forward_loss_wt

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.optimize = self.optimizer.minimize(self.predloss)
        # self.optimize = self.optimizer.minimize(self.predictor.forwardloss)  # when no feature encoding is used!
        self.cnt = 1

        self.sess2 = tf.Session()
        self.sess2.run(tf.global_variables_initializer())
        all_variables = tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)
        self.saver = tf.train.Saver(var_list=[v for v in all_variables if "Variab" not in v.name and "beta" not in v.name])

    def training(self, state_vec, prev_state_vec, action_1hot):
        _, predictionloss = self.sess2.run([self.optimize, self.predloss],
                                           feed_dict={self.predictor.s1: prev_state_vec,
                                           self.predictor.s2: state_vec,
                                           self.predictor.asample: action_1hot})
        return predictionloss

    def reward(self, s1, s2, asample):
        error = self.sess2.run(self.predictor.forwardloss,
                         {self.predictor.s1: [s1], self.predictor.s2: [s2], self.predictor.asample: [asample]})
        return error

    def inv_loss(self, s1, s2, asample):
        predloss, invloss = self.sess2.run([self.predloss, self.predictor.invloss],
                               {self.predictor.s1: [s1], self.predictor.s2: [s2], self.predictor.asample: [asample]})
        return predloss, invloss

    def predictedstate(self, s1, s2, asample):
        pred, orig = self.sess2.run([self.predictor.predstate, self.predictor.origstate],
                                    {self.predictor.s1: [s1], self.predictor.s2: [s2],
                                     self.predictor.asample: [asample]})
        return pred, orig

    def load_curiosity(self, load_filename):
        self.saver.restore(self.sess2, load_filename)
        print('Curiosity model has successfully loaded.')

    def save_ICM(self, save_filename):
        self.saver.save(self.sess2, save_filename)
        print('Curiosity model saved.')