
import tensorflow as tf
import model_prediction_curiosity as mpc
import os
from constants_prediction_curiosity import constants
import numpy as np

num_iterations = 5
learning_rate = 0.001
predictor = mpc.StateActionPredictor(268, 16, designHead='pydial')  # num belivestates, num actions

predloss = constants['PREDICTION_LR_SCALE'] * (
                                                predictor.invloss * (1 - constants['FORWARD_LOSS_WT']) +
                                                predictor.forwardloss * constants['FORWARD_LOSS_WT'])

optimizer = tf.train.AdamOptimizer(learning_rate)
optimize = optimizer.minimize(predloss)
# predgrads = tf.gradients(predloss * 20.0, predictor.var_list)  # change constant
# predgrads, _ = tf.clip_by_global_norm(predgrads, constants['GRAD_NORM_CLIP'])
# pred_grads_and_vars = list(zip(predgrads, predictor.var_list))
#
# optimize = optimizer.apply_gradients(pred_grads_and_vars)

if not os.path.exists('_curiosity_model/pretrg_model/'):
    os.mkdir('_curiosity_model/pretrg_model/')


def read_data1():
    sys_act = []
    turn = []
    with open('_curiosity_model/pretrg_data/2018-07-05_15:15:36', 'r') as d:  #TODO name of pre trg file changing not hard coded...
        for line in d:
            info = line.split(' ')
            turn.append(int(info[1]))
            sys_act.append(int(info[3]))
    return turn, sys_act


def read_data2():
    state = []
    prev_state = []
    with open('_curiosity_model/pretrg_data/prev_state2018-07-05_15:15:36', 'r') as d:
        for line in d:
            info = line.split(' ')
            prev_state.append(info)
    with open('_curiosity_model/pretrg_data/state2018-07-05_15:15:36', 'r') as d2:
        for line in d2:
            info = line.split(' ')
            state.append(info)
    return state, prev_state


sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

# read data from files
t, a = read_data1()
a = np.eye(16, 16)[a]  # convert to one-hot
s, _s = read_data2()

usable_len = len(t)/64*64
batch_num = usable_len/64

#todo : print evolution of prediction error and loss

loss = []
# # check if prev state vec is correct:
# if s[:-1] == _s[1:]:
#     print 'works'

for i in range(num_iterations):
    for batch in range(batch_num):
        # select batch for trg
        prev_state_vec = _s[batch * 64:(batch + 1) * 64]
        state_vec = s[batch * 64:(batch + 1) * 64]
        action_1hot = a[batch * 64:(batch + 1) * 64]
        _, predictionloss = sess.run([optimize, predloss],
                                     feed_dict={predictor.s1: prev_state_vec,
                                                predictor.s2: state_vec,
                                                predictor.asample: action_1hot})
        print predictionloss
        loss.append(predictionloss)
    saver.save(sess, '_curiosity_model/pretrg_model/trained_curiosity1')