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
ErrorSimulator.py - error simulation 
===============================================

Copyright CUED Dialogue Systems Group 2015 - 2017

.. seealso:: CUED Imports/Dependencies: 

    import :mod:`usersimulator.ConfidenceScorer` |.|
    import :mod:`usersimulator.ErrorModel` |.|
    import :mod:`utils.DiaAct` |.|
    import :mod:`utils.ContextLogger`

************************

''' 

__author__ = "cued_dialogue_systems_group"

import ConfidenceScorer
import ErrorModel
from utils import DiaAct
from utils import ContextLogger
logger = ContextLogger.getLogger('')



class DomainsErrorSimulator(object):
    def __init__(self, domainString, conf_scorer_name, error_rate):
        """
        Single domain error simulation module. Operates on semantic acts.
        :param: (str) conf_scorer_name
        :returns None:
        """
        self.em = ErrorModel.EM(domainString)
        self._set_confidence_scorer(conf_scorer_name)
        self._set_error_rate(error_rate)
    
    def _set_confidence_scorer(self, conf_scorer_name):
        conf_scorer_name = conf_scorer_name.lower()
        logger.info('Confidence scorer: %s' % conf_scorer_name)
        if conf_scorer_name == 'additive':
            self.confScorer = ConfidenceScorer.AdditiveConfidenceScorer(False, False)
        elif conf_scorer_name == 'dstc2':
            self.confScorer = ConfidenceScorer.DSTC2ConfidenceScorer()
        else:
            logger.warning('Invalid confidence scorer: %s. Using additive scorer.' % conf_scorer_name)
            self.confScorer = ConfidenceScorer.AdditiveConfidenceScorer(False, False)
        return
    
    def _set_error_rate(self, r):
        """Sets semantic error rate in :class:`ErrorModel` member

        :param: (int) semantic error rate
        :returns None: 
        """
        self.em.setErrorRate(r)
        
    def confuse_act(self, last_user_act):
        """Clean act in --> Confused act out. 

        :param: (str) simulated users semantic action
        :returns (list) of confused user acts.
        """
        uact = DiaAct.DiaActWithProb(last_user_act)
        n_best = self.em.getNBest(uact)
        n_best = self.confScorer.assignConfScores(n_best)
        
        # Normalise confidence scores
        dSum = 0.0
        for h in n_best:
            dSum += h.P_Au_O
        for h in n_best:
            h.P_Au_O /= dSum
        
        return n_best


#END OF FILE
