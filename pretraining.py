
'''
   This script is to pre-train a belief-state prediction model.
   This model then can be used in order to use belief-state prediction error as curiosity rewards.
'''

import tensorflow as tf
import model_prediction_curiosity as mpc
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# settings             **before running: make sure model_name is specified to not accidentally overwrite data**
model_name = ''  # name for new model: specify!
num_actions = 16
num_belief_states = 268
num_iterations = 3
learning_rate = 0.001
forward_loss_wt = 0.2
feature_size = 200
# file names pre-trg data: fill out!
action_pre_trg = ''
state_pre_trg = ''
prevstate_pre_trg = ''


# read actions and turns from pretrg data file
def read_data1(filename):
    sys_act = []
    turn = []
    with open(filename, 'r') as d:
        for line in d:
            info = line.split(' ')
            turn.append(int(info[1]))
            sys_act.append(int(info[3]))
    return turn, sys_act


# read state and prev_state from pretrg data file
def read_data2(filename_ps, filename_s):
    state = []
    prev_state = []
    with open(filename_ps, 'r') as d:
        for line in d:
            info = line.split(' ')
            prev_state.append(info)
    with open(filename_s, 'r') as d2:
        for line in d2:
            info = line.split(' ')
            state.append(info)
    return state, prev_state


def unison_shuffled_copies(vec1, vec2, vec3, vec4):
    assert len(vec1) == len(vec4)
    p = np.random.permutation(len(vec1))
    return np.array(vec1)[p], np.array(vec2)[p], np.array(vec3)[p], np.array(vec4)[p]


with tf.variable_scope('curiosity'):
    predictor = mpc.StateActionPredictor(num_belief_states, num_actions, designHead='pydial', feature_size=feature_size)
    predloss = predictor.invloss * (1 - forward_loss_wt) + predictor.forwardloss * forward_loss_wt

optimizer = tf.train.AdamOptimizer(learning_rate)
optimize = optimizer.minimize(predloss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

# read data from files
t, a = read_data1(action_pre_trg)
a = np.eye(16, 16)[a]  # convert to one-hot
s, _s = read_data2(prevstate_pre_trg, state_pre_trg)

# shuffle vectors
t, a, s, _s = unison_shuffled_copies(t, a, s, _s)

batch_num = len(t)/64

# initialize
loss = []
inverseloss = []
forwardloss = []

# # check if prev state vec is correct:
# if s[:-1] == _s[1:]:
#     print 'works'

if not os.path.exists('_curiosity_model/pretrg_model/'):
    os.mkdir('_curiosity_model/pretrg_model/')

# #  uncomment to train pre-trained model further
# saver.restore(sess, "_curiosity_model/pretrg_model/" + model_name)
# print("Successfully loaded:_curiosity_model/pretrg_model/" + model_name)

for i in range(num_iterations):
    for batch in range(batch_num):
        # select batch for trg
        prev_state_vec = _s[batch * 64:(batch + 1) * 64]
        state_vec = s[batch * 64:(batch + 1) * 64]
        action_1hot = a[batch * 64:(batch + 1) * 64]
        _, predictionloss, forloss, invloss = sess.run([optimize, predloss, predictor.forwardloss, predictor.invloss],
                                                       feed_dict={predictor.s1: prev_state_vec, predictor.s2: state_vec,
                                                                  predictor.asample: action_1hot})
        # if batch % 5 == 0:
        #     print predictionloss
        loss.append(predictionloss)
        inverseloss.append(invloss)
        forwardloss.append(forloss)

    t, a, s, _s = unison_shuffled_copies(t, a, s, _s)  # shuffle vectors

saver.save(sess, '_curiosity_model/pretrg_model/trained_curiosity_' + model_name + '_' + str(feature_size))

plt.plot(loss, label='prediction_loss')
plt.plot(inverseloss, label='inverse_loss')
plt.plot(forwardloss, label='forward_loss')
plt.legend()
plt.ylabel('Prediction error/ Loss')
plt.xlabel('number of batches')
plt.savefig('_plots/pretraining_loss_' + model_name + '_' + str(feature_size) + '.png', bbox_inches='tight')


# # uncomment if needed for experiments
# def curiosity_reward(s1, s2, asample):
#     error = sess.run(predictor.forwardloss,
#                           {predictor.s1: [s1], predictor.s2: [s2], predictor.asample: [asample]})
#     return error
#
# bonus = curiosity_reward(_s[13],s[13],action[13])
# print(bonus)
