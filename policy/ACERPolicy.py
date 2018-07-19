###############################################################################
# PyDial: Multi-domain Statistical Spoken Dialogue System Software
###############################################################################
#
# Copyright 2015 - 2018
# Cambridge University Engineering Department Dialogue Systems Group
#
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
###############################################################################

'''
ACERPolicy.py - Sample Efficient Actor Critic with Experience Replay
==================================================

Copyright CUED Dialogue Systems Group 2015 - 2017

The implementation of the sample efficient actor critic with truncated importance sampling with bias correction,
the trust region policy optimization method and RETRACE-like multi-step estimation of the value function.
The parameters ACERPolicy.c, ACERPolicy.alpha, ACERPolicy.
The details of the implementation can be found here: https://arxiv.org/abs/1802.03753

See also:
https://arxiv.org/abs/1611.01224
https://arxiv.org/abs/1606.02647

.. seealso:: CUED Imports/Dependencies:

    import :class:`Policy`
    import :class:`utils.ContextLogger`



************************

'''
import copy
import os
import json
import numpy as np
import scipy
import scipy.signal
import cPickle as pickle
import random
import utils
from utils.Settings import config as cfg
from utils import ContextLogger, DiaAct
from curiosity_module import Curious

import ontology.FlatOntologyManager as FlatOnt
import tensorflow as tf
from DRL.replay_buffer_episode_acer import ReplayBufferEpisode
from DRL.replay_prioritised_episode import ReplayPrioritisedEpisode
import DRL.utils as drlutils
import DRL.acer as acer
import Policy
import SummaryAction
from Policy import TerminalAction, TerminalState

logger = utils.ContextLogger.getLogger('')


# --- for flattening the belief --- #
def flatten_belief(belief, domainUtil, merge=False):
    belief = belief.getDomainState(domainUtil.domainString)
    if isinstance(belief, TerminalState):
        if domainUtil.domainString == 'CamRestaurants':
            return [0] * 268
        elif domainUtil.domainString == 'CamHotels':
            return [0] * 111
        elif domainUtil.domainString == 'SFRestaurants':
            return [0] * 633
        elif domainUtil.domainString == 'SFHotels':
            return [0] * 438
        elif domainUtil.domainString == 'Laptops11':
            return [0] * 257
        elif domainUtil.domainString == 'TV':
            return [0] * 188

    policyfeatures = ['full', 'method', 'discourseAct', 'requested', \
                      'lastActionInformNone', 'offerHappened', 'inform_info']

    flat_belief = []
    for feat in policyfeatures:
        add_feature = []
        if feat == 'full':
            # for slot in self.sorted_slots:
            for slot in domainUtil.ontology['informable']:
                for value in domainUtil.ontology['informable'][slot]:  # + ['**NONE**']:
                    add_feature.append(belief['beliefs'][slot][value])

                # pfb30 11.03.2017
                try:
                    add_feature.append(belief['beliefs'][slot]['**NONE**'])
                except:
                    add_feature.append(0.)  # for NONE
                try:
                    add_feature.append(belief['beliefs'][slot]['dontcare'])
                except:
                    add_feature.append(0.)  # for dontcare

        elif feat == 'method':
            add_feature = [belief['beliefs']['method'][method] for method in domainUtil.ontology['method']]
        elif feat == 'discourseAct':
            add_feature = [belief['beliefs']['discourseAct'][discourseAct]
                           for discourseAct in domainUtil.ontology['discourseAct']]
        elif feat == 'requested':
            add_feature = [belief['beliefs']['requested'][slot] \
                           for slot in domainUtil.ontology['requestable']]
        elif feat == 'lastActionInformNone':
            add_feature.append(float(belief['features']['lastActionInformNone']))
        elif feat == 'offerHappened':
            add_feature.append(float(belief['features']['offerHappened']))
        elif feat == 'inform_info':
            add_feature += belief['features']['inform_info']
        else:
            logger.error('Invalid feature name in config: ' + feat)

        flat_belief += add_feature

    return flat_belief


# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


class ACERPolicy(Policy.Policy):
    '''
    Derived from :class:`Policy`
    '''
    def __init__(self, in_policy_file, out_policy_file, domainString='CamRestaurants', is_training=False):
        super(ACERPolicy, self).__init__(domainString, is_training)

        tf.reset_default_graph()

        self.in_policy_file = in_policy_file
        self.out_policy_file = out_policy_file
        self.is_training = is_training
        self.accum_belief = []
        self.prev_state_check = None

        self.domainString = domainString
        self.domainUtil = FlatOnt.FlatDomainOntology(self.domainString)

        # parameter settings

        if 0:  # cfg.has_option('dqnpolicy', 'n_in'): #ic304: this was giving me a weird error, disabled it until i can check it deeper
            self.n_in = cfg.getint('dqnpolicy', 'n_in')
        else:
            self.n_in = self.get_n_in(domainString)

        self.actor_lr = 0.0001
        if cfg.has_option('dqnpolicy', 'actor_lr'):
            self.actor_lr = cfg.getfloat('dqnpolicy', 'actor_lr')

        self.critic_lr = 0.001
        if cfg.has_option('dqnpolicy', 'critic_lr'):
            self.critic_lr = cfg.getfloat('dqnpolicy', 'critic_lr')

        self.delta = 1.
        if cfg.has_option('dqnpolicy', 'delta'):
            self.delta = cfg.getfloat('dqnpolicy', 'delta')

        self.alpha = 0.99
        if cfg.has_option('dqnpolicy', 'beta'):
            self.alpha = cfg.getfloat('dqnpolicy', 'beta')

        self.c = 10.
        if cfg.has_option('dqnpolicy', 'is_threshold'):
            self.c = cfg.getfloat('dqnpolicy', 'is_threshold')

        self.randomseed = 1234
        if cfg.has_option('GENERAL', 'seed'):
            self.randomseed = cfg.getint('GENERAL', 'seed')

        self.gamma = 0.99
        if cfg.has_option('dqnpolicy', 'gamma'):
            self.gamma = cfg.getfloat('dqnpolicy', 'gamma')

        self.regularisation = 'l2'
        if cfg.has_option('dqnpolicy', 'regularisation'):
            self.regularisation = cfg.get('dqnpolicy', 'regularisation')

        self.learning_rate = 0.001
        if cfg.has_option('dqnpolicy', 'learning_rate'):
            self.learning_rate = cfg.getfloat('dqnpolicy', 'learning_rate')

        self.exploration_type = 'e-greedy' # Boltzman
        if cfg.has_option('dqnpolicy', 'exploration_type'):
            self.exploration_type = cfg.get('dqnpolicy', 'exploration_type')

        self.episodeNum = 1000
        if cfg.has_option('dqnpolicy', 'episodeNum'):
            self.episodeNum = cfg.getfloat('dqnpolicy', 'episodeNum')

        self.maxiter = 5000
        if cfg.has_option('dqnpolicy', 'maxiter'):
            self.maxiter = cfg.getfloat('dqnpolicy', 'maxiter')

        self.epsilon = 1
        if cfg.has_option('dqnpolicy', 'epsilon'):
            self.epsilon = cfg.getfloat('dqnpolicy', 'epsilon')

        self.epsilon_start = 1
        if cfg.has_option('dqnpolicy', 'epsilon_start'):
            self.epsilon_start = cfg.getfloat('dqnpolicy', 'epsilon_start')

        self.epsilon_end = 1
        if cfg.has_option('dqnpolicy', 'epsilon_end'):
            self.epsilon_end = cfg.getfloat('dqnpolicy', 'epsilon_end')

        self.priorProbStart = 1.0
        if cfg.has_option('dqnpolicy', 'prior_sample_prob_start'):
            self.priorProbStart = cfg.getfloat('dqnpolicy', 'prior_sample_prob_start')

        self.priorProbEnd = 0.1
        if cfg.has_option('dqnpolicy', 'prior_sample_prob_end'):
            self.priorProbEnd = cfg.getfloat('dqnpolicy', 'prior_sample_prob_end')

        self.policyfeatures = []
        if cfg.has_option('dqnpolicy', 'features'):
            logger.info('Features: ' + str(cfg.get('dqnpolicy', 'features')))
            self.policyfeatures = json.loads(cfg.get('dqnpolicy', 'features'))

        self.max_k = 5
        if cfg.has_option('dqnpolicy', 'max_k'):
            self.max_k = cfg.getint('dqnpolicy', 'max_k')

        self.learning_algorithm = 'drl'
        if cfg.has_option('dqnpolicy', 'learning_algorithm'):
            self.learning_algorithm = cfg.get('dqnpolicy', 'learning_algorithm')
            logger.info('Learning algorithm: ' + self.learning_algorithm)

        self.minibatch_size = 32
        if cfg.has_option('dqnpolicy', 'minibatch_size'):
            self.minibatch_size = cfg.getint('dqnpolicy', 'minibatch_size')

        self.capacity = 1000
        if cfg.has_option('dqnpolicy', 'capacity'):
            self.capacity = cfg.getint('dqnpolicy','capacity')

        self.replay_type = 'vanilla'
        if cfg.has_option('dqnpolicy', 'replay_type'):
            self.replay_type = cfg.get('dqnpolicy', 'replay_type')

        self.architecture = 'vanilla'
        if cfg.has_option('dqnpolicy', 'architecture'):
            self.architecture = cfg.get('dqnpolicy', 'architecture')

        self.q_update = 'single'
        if cfg.has_option('dqnpolicy', 'q_update'):
            self.q_update = cfg.get('dqnpolicy', 'q_update')

        self.h1_size = 130
        if cfg.has_option('dqnpolicy', 'h1_size'):
            self.h1_size = cfg.getint('dqnpolicy', 'h1_size')

        self.h2_size = 50
        if cfg.has_option('dqnpolicy', 'h2_size'):
            self.h2_size = cfg.getint('dqnpolicy', 'h2_size')

        self.save_step = 200
        if cfg.has_option('policy', 'save_step'):
            self.save_step = cfg.getint('policy', 'save_step')

        self.importance_sampling = 'soft'
        if cfg.has_option('dqnpolicy', 'importance_sampling'):
            self.importance_sampling = cfg.get('dqnpolicy', 'importance_sampling')

        self.train_iters_per_episode = 1
        if cfg.has_option('dqnpolicy', 'train_iters_per_episode'):
            self.train_iters_per_episode = cfg.getint('dqnpolicy', 'train_iters_per_episode')

        self.training_frequency = 2
        if cfg.has_option('dqnpolicy', 'training_frequency'):
            self.training_frequency = cfg.getint('dqnpolicy', 'training_frequency')

        # domain specific parameter settings (overrides general policy parameter settings)
        if cfg.has_option('dqnpolicy_'+domainString, 'n_in'):
            self.n_in = cfg.getint('dqnpolicy_'+domainString, 'n_in')

        if cfg.has_option('dqnpolicy_'+domainString, 'actor_lr'):
            self.actor_lr = cfg.getfloat('dqnpolicy_'+domainString, 'actor_lr')

        if cfg.has_option('dqnpolicy_'+domainString, 'critic_lr'):
            self.critic_lr = cfg.getfloat('dqnpolicy_'+domainString, 'critic_lr')

        if cfg.has_option('dqnpolicy_'+domainString, 'delta'):
            self.delta = cfg.getfloat('dqnpolicy_'+domainString, 'delta')

        if cfg.has_option('dqnpolicy_' + domainString, 'beta'):
            self.alpha = cfg.getfloat('dqnpolicy_' + domainString, 'beta')

        if cfg.has_option('dqnpolicy_' + domainString, 'is_threshold'):
            self.c = cfg.getfloat('dqnpolicy_' + domainString, 'is_threshold')

        if cfg.has_option('dqnpolicy_'+domainString, 'gamma'):
            self.gamma = cfg.getfloat('dqnpolicy_'+domainString, 'gamma')

        if cfg.has_option('dqnpolicy_'+domainString, 'regularisation'):
            self.regularisation = cfg.get('dqnpolicy_'+domainString, 'regulariser')

        if cfg.has_option('dqnpolicy_'+domainString, 'learning_rate'):
            self.learning_rate = cfg.getfloat('dqnpolicy_'+domainString, 'learning_rate')

        if cfg.has_option('dqnpolicy_'+domainString, 'exploration_type'):
            self.exploration_type = cfg.get('dqnpolicy_'+domainString, 'exploration_type')

        if cfg.has_option('dqnpolicy_'+domainString, 'episodeNum'):
            self.episodeNum = cfg.getfloat('dqnpolicy_'+domainString, 'episodeNum')

        if cfg.has_option('dqnpolicy_'+domainString, 'maxiter'):
            self.maxiter = cfg.getfloat('dqnpolicy_'+domainString, 'maxiter')

        if cfg.has_option('dqnpolicy_'+domainString, 'epsilon'):
            self.epsilon = cfg.getfloat('dqnpolicy_'+domainString, 'epsilon')

        if cfg.has_option('dqnpolicy_'+domainString, 'epsilon_start'):
            self.epsilon_start = cfg.getfloat('dqnpolicy_'+domainString, 'epsilon_start')

        if cfg.has_option('dqnpolicy_'+domainString, 'epsilon_end'):
            self.epsilon_end = cfg.getfloat('dqnpolicy_'+domainString, 'epsilon_end')

        if cfg.has_option('dqnpolicy_'+domainString, 'prior_sample_prob_start'):
            self.priorProbStart = cfg.getfloat('dqnpolicy_'+domainString, 'prior_sample_prob_start')

        if cfg.has_option('dqnpolicy_'+domainString, 'prior_sample_prob_end'):
            self.priorProbEnd = cfg.getfloat('dqnpolicy_'+domainString, 'prior_sample_prob_end')

        if cfg.has_option('dqnpolicy_'+domainString, 'features'):
            logger.info('Features: ' + str(cfg.get('dqnpolicy_'+domainString, 'features')))
            self.policyfeatures = json.loads(cfg.get('dqnpolicy_'+domainString, 'features'))

        if cfg.has_option('dqnpolicy_'+domainString, 'max_k'):
            self.max_k = cfg.getint('dqnpolicy_'+domainString, 'max_k')

        if cfg.has_option('dqnpolicy_'+domainString, 'learning_algorithm'):
            self.learning_algorithm = cfg.get('dqnpolicy_'+domainString, 'learning_algorithm')
            logger.info('Learning algorithm: ' + self.learning_algorithm)

        if cfg.has_option('dqnpolicy_'+domainString, 'minibatch_size'):
            self.minibatch_size = cfg.getint('dqnpolicy_'+domainString, 'minibatch_size')

        if cfg.has_option('dqnpolicy_'+domainString, 'capacity'):
            self.capacity = cfg.getint('dqnpolicy_'+domainString,'capacity')

        if cfg.has_option('dqnpolicy_'+domainString, 'replay_type'):
            self.replay_type = cfg.get('dqnpolicy_'+domainString, 'replay_type')

        if cfg.has_option('dqnpolicy_'+domainString, 'architecture'):
            self.architecture = cfg.get('dqnpolicy_'+domainString, 'architecture')

        if cfg.has_option('dqnpolicy_'+domainString, 'q_update'):
            self.q_update = cfg.get('dqnpolicy_'+domainString, 'q_update')

        if cfg.has_option('dqnpolicy_'+domainString, 'h1_size'):
            self.h1_size = cfg.getint('dqnpolicy_'+domainString, 'h1_size')

        if cfg.has_option('dqnpolicy_'+domainString, 'h2_size'):
            self.h2_size = cfg.getint('dqnpolicy_'+domainString, 'h2_size')

        if cfg.has_option('policy_' + domainString, 'save_step'):
            self.save_step = cfg.getint('policy_' + domainString, 'save_step')

        if cfg.has_option('dqnpolicy_'+domainString, 'importance_sampling'):
            self.importance_sampling = cfg.get('dqnpolicy_'+domainString, 'importance_sampling')

        if cfg.has_option('dqnpolicy_' + domainString, 'train_iters_per_episode'):
            self.train_iters_per_episode = cfg.getint('dqnpolicy_' + domainString, 'train_iters_per_episode')

        if cfg.has_option('dqnpolicy_'+domainString, 'training_frequency'):
            self.training_frequency = cfg.getint('dqnpolicy_'+domainString, 'training_frequency')

        self.curiosityreward = False
        if cfg.has_option('eval', 'curiosityreward'):
            self.curiosityreward = cfg.getboolean('eval', 'curiosityreward')


        self.episode_ct = 0

        self.episode_ave_max_q = []
        self.mu_prob = 0.  # behavioral policy

        #os.environ["CUDA_VISIBLE_DEVICES"]=""

        # init session
        self.sess = tf.Session()

        with tf.device("/cpu:0"):

            np.random.seed(self.randomseed)
            tf.set_random_seed(self.randomseed)

            # initialise an replay buffer
            if self.replay_type == 'vanilla':
                self.episodes[self.domainString] = ReplayBufferEpisode(self.capacity, self.minibatch_size, self.randomseed)
            elif self.replay_type == 'prioritized':
                self.episodes[self.domainString] = ReplayPrioritisedEpisode(self.capacity, self.minibatch_size, self.randomseed)
            #replay_buffer = ReplayBuffer(self.capacity, self.randomseed)
            #self.episodes = []
            self.samplecount = 0
            self.episodecount = 0

            # construct the models
            self.state_dim = self.n_in
            self.summaryaction = SummaryAction.SummaryAction(domainString)
            self.action_dim = len(self.summaryaction.action_names)
            action_bound = len(self.summaryaction.action_names)
            self.stats = [0 for _ in range(self.action_dim)]
            self.global_mu = [0. for _ in range(self.action_dim)]

            self.acer = acer.ACERNetwork(self.sess, self.state_dim, self.action_dim, \
                self.critic_lr, self.delta, self.c, self.alpha, self.h1_size, self.h2_size, self.is_training)

            # when all models are defined, init all variables
            init_op = tf.global_variables_initializer()
            self.sess.run(init_op)

            self.loadPolicy(self.in_policy_file)
            print 'loaded replay size: ', self.episodes[self.domainString].size()

            self.curiosityFunctions = Curious()
            #self.acer.update_target_network()

    def get_n_in(self, domain_string):
        if domain_string == 'CamRestaurants':
            return 268
        elif domain_string == 'CamHotels':
            return 111
        elif domain_string == 'SFRestaurants':
            return 636
        elif domain_string == 'SFHotels':
            return 438
        elif domain_string == 'Laptops6':
            return 268 # ic340: this is wrong
        elif domain_string == 'Laptops11':
            return 257
        elif domain_string is 'TV':
            return 188
        else:
            print 'DOMAIN {} SIZE NOT SPECIFIED, PLEASE DEFINE n_in'.format(domain_string)

    def get_n_in(self, domain_string):
        if domain_string == 'CamRestaurants':
            return 268
        elif domain_string == 'CamHotels':
            return 111
        elif domain_string == 'SFRestaurants':
            return 636
        elif domain_string == 'SFHotels':
            return 438
        elif domain_string == 'Laptops6':
            return 268 # ic340: this is wrong
        elif domain_string == 'Laptops11':
            return 257
        elif domain_string is 'TV':
            return 188
        else:
            print 'DOMAIN {} SIZE NOT SPECIFIED, PLEASE DEFINE n_in'.format(domain_string)

    def act_on(self, state, hyps=None):
        if self.lastSystemAction is None and self.startwithhello:
            systemAct, nextaIdex, mu, mask = 'hello()', -1, None, None
        else:
            systemAct, nextaIdex, mu, mask = self.tion(state)
        self.lastSystemAction = systemAct
        self.summaryAct = nextaIdex
        self.prev_mu = mu
        self.prev_mask = mask
        self.prevbelief = state

        systemAct = DiaAct.DiaAct(systemAct)
        return systemAct

    def record(self, reward, domainInControl=None, weight=None, state=None, action=None):
        if domainInControl is None:
            domainInControl = self.domainString
        if self.actToBeRecorded is None:
            #self.actToBeRecorded = self.lastSystemAction
            self.actToBeRecorded = self.summaryAct

        if state is None:
            state = self.prevbelief
        if action is None:
            action = self.actToBeRecorded
        mu_weight = self.prev_mu
        mask = self.prev_mask

        cState, cAction = self.convertStateAction(state, action)

        # normalising total return to -1~1
        #reward /= 40.0
        reward /= 20.0
        """
        reward = float(reward+10.0)/40.0
        """
        value = self.acer.predict_value([cState], [mask])

        if self.replay_type == 'vanilla':
            self.episodes[domainInControl].record(state=cState, \
                    state_ori=state, action=cAction, reward=reward, value=value[0], distribution=mu_weight, mask=mask)
        elif self.replay_type == 'prioritized':
            self.episodes[domainInControl].record(state=cState, \
                    state_ori=state, action=cAction, reward=reward, value=value[0], distribution=mu_weight, mask=mask)

        self.actToBeRecorded = None
        self.samplecount += 1
        return

    def finalizeRecord(self, reward, domainInControl=None):
        if domainInControl is None:
            domainInControl = self.domainString
        if self.episodes[domainInControl] is None:
            logger.warning("record attempted to be finalized for domain where nothing has been recorded before")
            return

        #print 'Episode Avg_Max_Q', float(self.episode_ave_max_q)/float(self.episodes[domainInControl].size())
        #print 'Episode Avg_Max_Q', np.mean(self.episode_ave_max_q)
        #print self.stats

        # normalising total return to -1~1
        reward /= 20.0

        terminal_state, terminal_action = self.convertStateAction(TerminalState(), TerminalAction())
        value = 0.0 # not effect on experience replay

        def calculate_discountR_advantage(r_episode, v_episode):
            #########################################################################
            # Here we take the rewards and values from the rollout, and use them to
            # generate the advantage and discounted returns.
            # The advantage function uses "Generalized Advantage Estimation"
            bootstrap_value = 0.0
            self.r_episode_plus = np.asarray(r_episode + [bootstrap_value])
            discounted_r_episode = discount(self.r_episode_plus,self.gamma)[:-1]
            self.v_episode_plus = np.asarray(v_episode + [bootstrap_value])
            advantage = r_episode + self.gamma * self.v_episode_plus[1:] - self.v_episode_plus[:-1]
            advantage = discount(advantage,self.gamma)
            #########################################################################
            return discounted_r_episode, advantage

        if self.replay_type == 'vanilla':
            self.episodes[domainInControl].record(state=terminal_state, \
                    state_ori=TerminalState(), action=terminal_action, reward=reward, value=value, terminal=True, distribution=None)
        elif self.replay_type == 'prioritized':
            episode_r, episode_v = self.episodes[domainInControl].record_final_and_get_episode(state=terminal_state, \
                                                                                               state_ori=TerminalState(),
                                                                                               action=terminal_action,
                                                                                               reward=reward,
                                                                                               value=value)

            # TD_error is a list of td error in the current episode
            _, TD_error = calculate_discountR_advantage(episode_r, episode_v)
            episodic_TD = np.mean(np.absolute(TD_error))
            print 'episodic_TD'
            print episodic_TD
            self.episodes[domainInControl].insertPriority(episodic_TD)

        return

    def convertStateAction(self, state, action):
        if isinstance(state, TerminalState):
            if self.domainUtil.domainString == 'CamRestaurants':
                return [0] * 268, action
            elif self.domainUtil.domainString == 'CamHotels':
                return [0] * 111, action
            elif self.domainUtil.domainString == 'SFRestaurants':
                return [0] * 633, action
            elif self.domainUtil.domainString == 'SFHotels':
                return [0] * 438, action
            elif self.domainUtil.domainString == 'Laptops11':
                return [0] * 257, action
            elif self.domainUtil.domainString == 'TV':
                return [0] * 188, action
        else:
            flat_belief = flatten_belief(state, self.domainUtil)
            self.prev_state_check = flat_belief

            return flat_belief, action

    def tion(self, beliefstate):
        '''
        select next action

        :param beliefstate:
        :param hyps:
        :returns: (int) next summarye action
        '''
        beliefVec = flatten_belief(beliefstate, self.domainUtil)

        execMask = self.summaryaction.getExecutableMask(beliefstate, self.lastSystemAction)
        #execMask = np.zeros(self.action_dim)

        def apply_mask(prob, maskval, baseline=9.99999975e-06):
            return prob if maskval == 0.0 else baseline # not quite 0.0 to avoid division by zero

        action_prob = self.acer.predict_policy(np.reshape(beliefVec, (1, len(beliefVec))),
                                               np.reshape(execMask, (1, len(execMask))))[0]

        if self.exploration_type == 'e-greedy' or not self.is_training:
            # epsilon greedy
            epsilon = self.epsilon if self.is_training else 0.
            eps_prob = [apply_mask(prob, admissible) for prob, admissible in zip(np.ones(len(execMask)), execMask)]
            eps_prob /= sum(eps_prob)

            #action_prob = [apply_mask(prob, admissible) for prob, admissible in zip(action_prob, execMask)]
            best_index = np.argmax(action_prob)
            best_prob = [1. if i == best_index else 0. for i in range(len(action_prob))]

            action_prob = epsilon * np.array(eps_prob) + (1. - epsilon) * np.array(best_prob)

        elif self.exploration_type == 'standard':
            #action_prob = [apply_mask(prob, admissible) for prob, admissible in zip(action_prob, execMask)]
            print action_prob

        nextaIdex = np.random.choice(len(action_prob), p=action_prob / sum(action_prob))
        mu = action_prob / sum(action_prob)

        self.stats[nextaIdex] += 1
        self.global_mu[nextaIdex] += 1
        summaryAct = self.summaryaction.action_names[nextaIdex]
        beliefstate = beliefstate.getDomainState(self.domainUtil.domainString)
        masterAct = self.summaryaction.Convert(beliefstate, summaryAct, self.lastSystemAction)
        return masterAct, nextaIdex, mu, execMask

    def train(self):
        '''
        call this function when the episode ends
        '''
        USE_GLOBAL_MU = False
        self.episode_ct += 1

        if not self.is_training:
            logger.info("Not in training mode")
            return
        else:
            logger.info("Update acer policy parameters.")

        import time
        start = time.time()

        self.episodecount += 1
        logger.info("Sample Num so far: %s" % (self.samplecount))
        logger.info("Episode Num so far: %s" % (self.episodecount))
        #if True:
        if self.samplecount >= self.minibatch_size * 3 and self.episodecount % self.training_frequency == 0:
        # if self.episodecount >= self.minibatch_size  and self.episodecount % 2 == 0:
        # if self.episodecount >= self.minibatch_size * 3 and self.episodecount % 2 == 0:
        # if self.samplecount >= self.capacity and self.episodecount % 5 == 0:
            logger.info('start trainig...')

            for _ in range(self.train_iters_per_episode):

                if self.replay_type == 'vanilla' or self.replay_type == 'prioritized':
                    s_batch, s_ori_batch, a_batch, r_batch, s2_batch, s2_ori_batch, t_batch, idx_batch, v_batch, mu_policy, mask_batch = \
                        self.episodes[self.domainString].sample_batch()
                    if USE_GLOBAL_MU:
                        mu_sum = sum(self.global_mu)
                        mu_normalised = np.array([c / mu_sum for c in self.global_mu])
                        #print >> sys.stderr, len(mu_policy), len(mu_policy[0]), mu_policy[0][0]
                        mu_policy = [[mu_normalised for _ in range(len(mu_policy[i]))] for i in range(len(mu_policy))]
                else:
                    assert False  # not implemented yet

                discounted_r_batch = []
                advantage_batch = []

                def calculate_discountR_advantage(r_episode, v_episode):
                    #########################################################################
                    # Here we take the rewards and values from the rolloutv, and use them to
                    # generate the advantage and discounted returns.
                    # The advantage function uses "Generalized Advantage Estimation"
                    bootstrap_value = 0.0
                    # r_episode rescale by rhos?
                    self.r_episode_plus = np.asarray(r_episode + [bootstrap_value])
                    discounted_r_episode = discount(self.r_episode_plus, self.gamma)[:-1]
                    self.v_episode_plus = np.asarray(v_episode + [bootstrap_value])
                    # change sth here
                    advantage = r_episode + self.gamma * self.v_episode_plus[1:] - self.v_episode_plus[:-1]
                    advantage = discount(advantage, self.gamma)
                    #########################################################################
                    return discounted_r_episode, advantage

                if self.replay_type == 'prioritized':
                    for item_r, item_v, item_idx in zip(r_batch, v_batch, idx_batch):
                        # r, a = calculate_discountR_advantage(item_r, np.concatenate(item_v).ravel().tolist())
                        r, a = calculate_discountR_advantage(item_r, item_v)

                        # flatten nested numpy array and turn it into list
                        discounted_r_batch += r.tolist()
                        advantage_batch += a.tolist()

                        # update the sum-tree
                        # update the TD error of the samples (episode) in the minibatch
                        episodic_TD_error = np.mean(np.absolute(a))
                        self.episodes[self.domainString].update(item_idx, episodic_TD_error)
                else:
                    for item_r, item_v in zip(r_batch, v_batch):
                        # r, a = calculate_discountR_advantage(item_r, np.concatenate(item_v).ravel().tolist())
                        r, a = calculate_discountR_advantage(item_r, item_v)

                        # flatten nested numpy array and turn it into list
                        discounted_r_batch += r.tolist()
                        advantage_batch += a.tolist()

                batch_size = len(s_batch)

                a_batch_one_hot = np.eye(self.action_dim)[np.concatenate(a_batch, axis=0).tolist()]

                if self.curiosityreward:
                    curiosity_loss = \
                        self.curiosityFunctions.training(np.concatenate(np.array(s2_batch), axis=0).tolist(),
                                                         np.concatenate(np.array(s_batch), axis=0).tolist(),
                                                         a_batch_one_hot)

                loss, entropy, optimize = \
                            self.acer.train(np.concatenate(np.array(s_batch), axis=0).tolist(), a_batch_one_hot,
                                            np.concatenate(np.array(mask_batch), axis=0).tolist(),
                                            np.concatenate(np.array(r_batch), axis=0).tolist(), s_batch, r_batch, self.gamma,
                                            np.concatenate(np.array(mu_policy), axis=0),
                                            discounted_r_batch, advantage_batch)

                ent, norm_loss = entropy/float(batch_size), loss/float(batch_size)

                #print 'normalised from %d episodes' %(batch_size)
                #print >> sys.stderr, 'entropy', ent
                #print >> sys.stderr, 'total loss', norm_loss

            #if self.episodecount % 1 == 0:
            #    self.acer.update_target_network()

            self.savePolicyInc()  # self.out_policy_file)

            end = time.time()
            #print >> sys.stderr, 'training took %s' % str(end - start)

    def savePolicy(self, FORCE_SAVE=False):
        """
        Does not use this, cause it will be called from agent after every episode.
        we want to save the policy only periodically.
        """
        pass

    def savePolicyInc(self, FORCE_SAVE=False):
        """
        save model and replay buffer
        """
        if self.episodecount % self.save_step == 0:
            #save_path = self.saver.save(self.sess, self.out_policy_file+'.ckpt')
            self.acer.save_network(self.out_policy_file+'.acer.ckpt')

            f = open(self.out_policy_file+'.episode', 'wb')
            for obj in [self.samplecount, self.episodes[self.domainString], self.global_mu]:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.close()
            #logger.info("Saving model to %s and replay buffer..." % save_path)

    def loadPolicy(self, filename):
        """
        load model and replay buffer
        """
        # load models
        self.acer.load_network(filename+'.acer.ckpt')

        # load replay buffer
        try:
            print 'laod from: ', filename
            f = open(filename+'.episode', 'rb')
            loaded_objects = []
            for i in range(2): # load nn params and collected data
                loaded_objects.append(pickle.load(f))
            self.samplecount = int(loaded_objects[0])
            self.episodes[self.domainString] = copy.deepcopy(loaded_objects[1])
            self.global_mu = loaded_objects[2]
            logger.info("Loading both model from %s and replay buffer..." % filename)
            f.close()
        except:
            logger.info("Loading only models...")

    def restart(self):
        self.summaryAct = None
        self.lastSystemAction = None
        self.prevbelief = None
        self.prev_mu = None
        self.prev_mask = None
        self.actToBeRecorded = None
        self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * float(self.episodeNum+self.episodecount) / float(self.maxiter)
        #print 'current eps', self.epsilon
        #self.episodes = dict.fromkeys(OntologyUtils.available_domains, None)
        #self.episodes[self.domainString] = ReplayBuffer(self.capacity, self.randomseed)
        self.episode_ave_max_q = []

#END OF FILE
