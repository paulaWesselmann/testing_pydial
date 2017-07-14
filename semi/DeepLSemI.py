###############################################################################
# PyDial: Multi-domain Statistical Spoken Dialogue System Software
###############################################################################
#
# Copyright 2015 - 2017
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
DeepLSemI.py - Deep Learning Semantic Decoder
====================================================================================

To use this in pydial, need to set "semitype = DLSemI" for a domain in the relevant interface config file
(in the current state it is the CamRestaurants domain)
See texthub_dl.cfg, which can be used for this purpose for texthub interface

.. seealso:: Exploiting Sentence and Context Representations in Deep Neural Models for Spoken Language Understanding
    https://arxiv.org/abs/1610.04120

.. seealso:: CUED Imports/Dependencies:

    import :mod:`semi.SemI` |.|
    import :mod:`dlsemi` |.|

.. seealso:: dlsemi_light:
    https://lmr46@bitbucket.org/lmr46/dlsemi_light.git

Important: you must install dlsemi_light as a package, in order to use this class

    `
'''


import os, sys
#logger = ContextLogger.getLogger('')
from semi import SemI
import dlsemi
from dlsemi import DLSemI
from dlsemi import test
from dlsemi import datasets
path = os.path.dirname(dlsemi.__file__)
os.sys.path.insert(1, path)
import math
import time
import RegexSemI
print sys.path

__author__ = "cued_dialogue_systems_group"

class DeepLSemI(SemI.SemI):

    def __init__(self):
        '''
        Initialise class and initialise RegexSemI to deal with classification errors
        :return:
        '''
        self.RSemI = RegexSemI.RegexSemI() # For goodbye and request alternatives in decode

        self.old_path = os.getcwd()
        path = os.path.dirname(dlsemi.__file__)
        os.chdir(path)

        self.datasets, self.nasrs, cl_name, exclude_name, static, context, window, sa_st, \
        slot_best_model, sl_name, val_static, val_context, val_window, val_sa_st, sv_init_model = DLSemI.readSemIConfig(\
            'config/dstc2.cfg', opts=None)
        DLSemI.load_wvectors_and_models(self.datasets)

        os.chdir(self.old_path)

        self.sys_act = [['welcome','--','--']]


    def decode(self, ASR_obs, sys_act=None, turn=None):
        '''
        Includes os.chdir to change directories from pydial root to the locally installed dlsemi package right before
        dlsemi is called. Directories are changed back to pydial root after prediction. This ensures all the required
        config and data files are accessed.
        :param ASR_obs:
        :param sys_act:
        :param turn:
        :return:
        '''


        if sys_act!=None:
            a, b = [sys_act.split('(', 1)[0], sys_act.split('(', 1)[1].split(')')[0]]
            self.sys_act.append([a, 'slot', b])

        #Check first general dialogue acts with Regular Expressions
        regexpred = self.decode_general_hypothesis(ASR_obs[0][0])

        if "bye()" in regexpred:
            return [("bye()", 1.0)]
        elif "reqalts()" in regexpred:
            return [("reqalts()", 1.0)]
        elif "affirm()" in regexpred:
            return [("affirm()",1.0)]
        elif "negate()"in regexpred:
            return [("negate()",1.0)]
        elif "hello()" in regexpred:
            return [("hello()",1.0)]
        else:
            old_path = os.getcwd()
            path = os.path.dirname(dlsemi.__file__)
            os.chdir(path)

            sentinfo = self.input_json(ASR_obs, self.sys_act, turn)

            before = int(round(time.time() * 1000))
            predictions = DLSemI.predict_sent(self.datasets, self.nasrs, sentinfo)

            after = int(round(time.time() * 1000))
            pred_dur = after - before
            print "prediction time: %d" % pred_dur # Time taken by DLSemI for prediction

            os.chdir(old_path)

            self.semActs = self.format_semi_output(predictions)

            return self.semActs


    def input_json(self, ASR_obs, sys_act, turn):
        '''
        Formats the incoming ASR_obs and sys_act into an input for DLSemI in JSON
        :param ASR_obs:
        :param sys_act:
        :param turn:
        :return:
        '''


        sentinfo = {}
        keys = [u'turn-id', u'asr-hyps', u'prevsysacts']

        asrhyps = []
        for obs in ASR_obs:
            asrhyps.append(dict([ (u'asr-hyp', unicode(obs[0])), (u'score', math.log(obs[1]))]))

        prevsysacts = []
        prevsysactskeys = [u'dact', u'slot', u'value']
        for act in sys_act:
            prevsysacts.append(dict([ (prevsysactskeys[n], act[n]) for n in range(3)]))

        # Ensures only last four sys_acts are used
        if len(prevsysacts) > 4:
            prevsysacts_lastfour = prevsysacts[-5:-1]
        else:
            prevsysacts_lastfour = prevsysacts

        values = [turn, asrhyps, prevsysacts_lastfour]

        for a in range(3):
            sentinfo[keys[a]] = values[a]

        return sentinfo


    def format_semi_output(self, dlsemiprediction):
        '''
        Transform the DLSemI output to make it compatible with cued-pydial system
        :param dlsemiprediction: output coming from DLSemI
        :return: DLSemI output in the required format for cued-pydial
        '''

        if not dlsemiprediction:
            prediction_clean = [('',1.0)]
        else:
            dact = dlsemiprediction['dact'][0]
            prediction_keys = [key for key, value in dlsemiprediction.items() if key not in ['dact']]
            prediction_string = []

            probability = dlsemiprediction['dact'][1]

            for key in prediction_keys:
                prediction_string.append('%s(%s=%s)' % (unicode(dact), unicode(key), unicode(dlsemiprediction[key][0])))
                probability = probability * dlsemiprediction[key][1]

            prediction_string = '|'.join(prediction_string)
            prediction_clean = [(prediction_string, probability)]

        return prediction_clean


    def decode_general_hypothesis(self, obs):
        '''
        Regular expressions for bye() and reqalts(), affirm and type
        :param obs: ASR hypothesis
        :return: RegexSemI recognised dialogue act
        '''
        self.RSemI.semanticActs = []

        self.RSemI._decode_reqalts(obs)
        self.RSemI._decode_bye(obs)
        self.RSemI._decode_type(obs)
        self.RSemI._decode_affirm(obs)

        return self.RSemI.semanticActs


#if __name__ == '__main__':
    #dls=DLSemI_test()
    #dls.decode([('I am looking for a chinese restaurant in the center',1.0)])