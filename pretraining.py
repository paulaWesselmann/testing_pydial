
import tensorflow as tf
import model_prediction_curiosity as mpc
import os
from constants_prediction_curiosity import constants
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

num_iterations = 5
learning_rate = 0.001
with tf.variable_scope('curiosity'):
    predictor = mpc.StateActionPredictor(268, 16, designHead='pydial')  # num belivestates, num actions

    predloss = predictor.invloss * (1 - constants['FORWARD_LOSS_WT']) + \
               predictor.forwardloss * constants['FORWARD_LOSS_WT']

optimizer = tf.train.AdamOptimizer(learning_rate)
optimize = optimizer.minimize(predloss)
# predgrads = tf.gradients(predloss * 20.0, predictor.var_list)  # change constant
# predgrads, _ = tf.clip_by_global_norm(predgrads, constants['GRAD_NORM_CLIP'])
# pred_grads_and_vars = list(zip(predgrads, predictor.var_list))
#
# optimize = optimizer.apply_gradients(pred_grads_and_vars)

if not os.path.exists('_curiosity_model/pretrg_model/'):
    os.mkdir('_curiosity_model/pretrg_model/')


def read_data1(filename):
    sys_act = []
    turn = []
    with open(filename, 'r') as d:
        for line in d:
            info = line.split(' ')
            turn.append(int(info[1]))
            sys_act.append(int(info[3]))
    return turn, sys_act


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


sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

# read data from files
t, a = read_data1('_curiosity_model/pretrg_data/dqn2018-07-18_15:34:09')  # 2018-07-05_15:15:36
a = np.eye(16, 16)[a]  # convert to one-hot
s, _s = read_data2('_curiosity_model/pretrg_data/dqn_prev_state2018-07-18_15:34:09',
                   '_curiosity_model/pretrg_data/dqn_state2018-07-18_15:34:09')  # prev_state2018-07-05_15:15:36

usable_len = len(t)/64*64
batch_num = usable_len/64

#todo : print evolution of prediction error and loss

loss = [] #todo why is loss starting at 2 and not 20 anyamore???
# # check if prev state vec is correct:
# if s[:-1] == _s[1:]:
#     print 'works'

# **********************uncomment to train pre-trained model more:**********************************************
saver.restore(sess, "_curiosity_model/pretrg_model/trained_curiosity100")
print("Successfully loaded:_curiosity_model/pretrg_model/trained_curiosity100")

# # to correct prev state error in data: first turn prev state = zeros
# i = 0
# for num in t:
#     if num == 1:
#         _s[i] = np.zeros(268)
#     i += 1


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
        if batch % 5 == 0:
            print predictionloss
        loss.append(predictionloss)

    saver.save(sess, '_curiosity_model/pretrg_model/trained_curiosity3')
    # 1, 100 : hdc, (error:)prev state for first not zeros?
    # 2: dqn, prev state for first turn zeros now
    # 3: 100+dqn, use pre trained 100 model and train with dqn data

# mpl.use('Agg')
plt.plot(loss)
plt.savefig('_plots/pretraining_loss_hcd100dqn.png', bbox_inches='tight')

# def curiosity_reward(s1, s2, asample):
#     error = sess.run(predictor.forwardloss,
#                           {predictor.s1: [s1], predictor.s2: [s2], predictor.asample: [asample]})
#     return error
#
#
# turn, action = read_data1('_curiosity_model/pretrg_data/2018-07-06_01:40:59')
# _s, s = read_data2('_curiosity_model/pretrg_data/prev_state2018-07-06_01:40:59',
#                    '_curiosity_model/pretrg_data/state2018-07-06_01:40:59')
# action = np.eye(16, 16)[action]
#
# # example dialogue:
# dialogue = range(61)
# bonus = []
# actions = []
# for i in dialogue:
#     bonus.append(curiosity_reward(_s[i], s[i], action[i]))
#     actions.append(np.where(action[i] == 1)[0][0])
#     # print('turn: '+str(turn[i])+' action: '+str(actions[i])+' curiosity: '+str(bonus[i])+' state: '+str(s[i]))
#
# plt.scatter(actions, bonus)
# plt.savefig('_plots/action_bonus2.png', bbox_inches='tight')

# #turn 1
# bonus1 = curiosity_reward(_s[0],s[0],action[0]) #13
# print(bonus1)
# #turn 2
# bonus1 = curiosity_reward(s[0],s[0],action[0])
# print(bonus1)
# #turn 2
# bonus1 = curiosity_reward(_s[12],s[12],action[12])
# print(bonus1)
#
# # turn 1
# bonus1 = curiosity_reward(_s[13],s[13],action[13])
# print(bonus1)
# #turn 2
# bonus1 = curiosity_reward(_s[14],s[14],action[14])
# print(bonus1)
# # turn 1
# bonus1 = curiosity_reward(_s[15],s[15],action[15])
# print(bonus1)
# #turn 2
# bonus1 = curiosity_reward(_s[16],s[16],action[16])
# print(bonus1)
# # turn 1
# bonus1 = curiosity_reward(_s[17],s[17],action[17])
# print(bonus1)
# #turn 2
# bonus1 = curiosity_reward(s[14],s[14],action[14])
# print(bonus1)


