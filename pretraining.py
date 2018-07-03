
import tensorflow as tf
import model_prediction_curiosity as mpc
import os
from constants_prediction_curiosity import constants
import numpy as np
# from datetime import datetime

#Pre_training() #todo call class and run trg

sys_act = []
state = []
prev_state = []
turn = []
num_iterations = 5
batch_num = 20
learning_rate = 0.001
predictor = mpc.StateActionPredictor(268, 16, designHead='pydial')

predloss = constants['PREDICTION_LR_SCALE'] * (
predictor.invloss * (1 - constants['FORWARD_LOSS_WT']) +
predictor.forwardloss * constants['FORWARD_LOSS_WT'])

optimizer = tf.train.AdamOptimizer(learning_rate)
optimize = optimizer.minimize(predloss) #todo not used? see same in model pred and dqn

predgrads = tf.gradients(predloss * 20.0, predictor.var_list)  # todo change constant
predgrads, _ = tf.clip_by_global_norm(predgrads, constants['GRAD_NORM_CLIP'])
pred_grads_and_vars = list(zip(predgrads, predictor.var_list))

optimize = optimizer.apply_gradients(pred_grads_and_vars)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

if not os.path.exists('_curiosity_model/pretrg_model/'):
    os.mkdir('_curiosity_model/pretrg_model/')


def read_data():
    with open('_curiosity_model/pretrg_data/', 'r') as d:  # todo add name of pretrg file!
        for line in d:
            info = line.split(' ')
            turn.append(info[1])
            sys_act.append(info[3])
            state.append(info[5])
            prev_state.append(info[7])
    return turn, sys_act, state, prev_state


def run_trg(prev_s, s, a):
    loss = []
    np.reshape(prev_s, (batch_num, -1))
    np.reshape(s, (batch_num, -1))
    np.reshape(a, (batch_num, -1))
    for i in range(num_iterations):
        for batch in range(batch_num):
            loss.append(train_curiosity(prev_s[batch], s[batch], a[batch]))
    return loss


# #TODO make multiple batches from data and use batch train multiple iterations of epoch!
# #make batches and how do i call stuff?
#
# #check linesplit are there spaces in list?


def train_curiosity(prev_state_vec, state_vec, action_1hot):
    _, predictionloss = sess.run([optimize, predloss],
                                      feed_dict={predictor.s1: prev_state_vec,
                                                 predictor.s2: state_vec,
                                                 predictor.asample: action_1hot})
    # date_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    saver.save(sess, '_curiosity_model/pretrg_model/trained_curiosity1')
    return predictionloss


turn, sys_act, state, prev_state = read_data()
run_trg(prev_state, state, sys_act)

#TODO inspect this file with debug mode