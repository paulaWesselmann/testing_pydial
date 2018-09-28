###############################################################################
# adapted from:
# Deepak Pathak, Pulkit Agrawal, Alexei A. Efros, Trevor Darrell
# University of California, Berkeley
# Curiosity-driven Exploration by Self-supervised Prediction

# added by Paula
###############################################################################

from __future__ import print_function
import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import os


def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


def flatten(x):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])


def cosineLoss(A, B, name):
    ''' A, B : (BatchSize, d) '''
    dotprod = tf.reduce_sum(tf.multiply(tf.nn.l2_normalize(A, 1), tf.nn.l2_normalize(B, 1)), 1)
    loss = 1-tf.reduce_mean(dotprod, name=name)
    return loss


def linear(x, size, name, initializer=None, bias_init=0):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        w = tf.get_variable("w", [x.get_shape()[1], size], initializer=initializer) # error in second turn, reuse variable?
        b = tf.get_variable("b", [size], initializer=tf.constant_initializer(bias_init)) #changed from: name+ "/b"
        # initialized now to fix error
    return tf.matmul(x, w) + b


def inverse_pydialHead(x, final_shape):  # does nothing so far!
    '''
                input: [None, 268]; output: [None, 1, 268];
    '''
    # print('Using inverse-pydial head design')
    # bs = tf.shape(x)[0]
    # print(x)
    return x


def pydialHead(x, layer2):  # todo: 200 is size of layer, make it var and enable easy change such as feat_size
    '''
            input: [None, 1, 268]; output: [None, ?];
    '''
    x = tf.nn.elu(linear(x, 200, 'fc', normalized_columns_initializer(0.01)))
    # print(x.get_shape())
    # x = flatten(x)
    # print(x.get_shape())
    # x = tf.nn.elu(x)
    return x


class StateActionPredictor(object):
    def __init__(self, ob_space, ac_space, designHead='pydial', feature_size=200, layer2=200):
        # input: s1,s2: : [None, h, w, ch] (usually ch=1 or 4) /pydial: [None, size]
        # asample: 1-hot encoding of sampled action from policy: [None, ac_space]

        self.layer2 = layer2
        if designHead == 'pydial':
            input_shape = [None, ob_space]
        else:
            input_shape = [None] + list(ob_space)

        self.s1 = phi1 = tf.placeholder(tf.float32, input_shape)
        self.s2 = phi2 = tf.placeholder(tf.float32, input_shape)
        self.asample = asample = tf.placeholder(tf.float32, [None, ac_space])

        # feature encoding: phi1, phi2: [None, LEN]
        size = feature_size  # 268 for full believstate
        if designHead == 'pydial':
            phi1 = pydialHead(phi1, self.layer2)
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                phi2 = pydialHead(phi2, self.layer2)
        else:
            print('So far "pydial" is the only available design head. Please check your configurations.')

        # inverse model: g(phi1,phi2) -> a_inv: [None, ac_space]
        g = tf.concat([phi1, phi2], 1)   # changed place of 1
        g = tf.nn.relu(linear(g, size, "g1", normalized_columns_initializer(0.01)))
        aindex = tf.argmax(asample, axis=1)  # aindex: [batch_size,]
        logits = linear(g, ac_space, "glast", normalized_columns_initializer(0.01))
        self.invloss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=aindex), name="invloss")
        self.ainvprobs = tf.nn.softmax(logits, axis=-1)

        # forward model: f(phi1,asample) -> phi2
        # Note: no backprop to asample of policy: it is treated as fixed for predictor training
        f = tf.concat([phi1, asample], 1)
        f = tf.nn.relu(linear(f, size, "f1", normalized_columns_initializer(0.01)))
        f = linear(f, phi1.get_shape()[1].value, "flast", normalized_columns_initializer(0.01))
        # self.forwardloss = 0.5 * tf.reduce_mean(tf.square(tf.subtract(f, phi2)), name='forwardloss')

        # self.forwardloss = 0.5 * tf.reduce_mean(tf.sqrt(tf.abs(tf.subtract(f, phi2))), name='forwardloss')
        self.forwardloss = cosineLoss(f, phi2, name='forwardloss')
        # self.forwardloss = self.forwardloss * 268.0  # lenFeatures=268. Factored out to make hyperparams not depend on it.

        # prediction and original
        self.predstate = f
        self.origstate = phi2

