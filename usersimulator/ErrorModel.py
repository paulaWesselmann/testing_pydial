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
ErrorModel.py - error simulation 
===============================================

Copyright CUED Dialogue Systems Group 2015 - 2017

 
**Relevant Config variables** [Default values]::

    [em]
    nbestsize = 1 
    confusionmodel = RandomConfusions
    nbestgeneratormodel = UniformNBestGenerator

.. seealso:: CUED Imports/Dependencies: 

    import :mod:`usersimulator.ConfusionModel` |.|
    import :mod:`usersimulator.NBestGenerator` |.|
    import :mod:`utils.Settings` |.|
    import :mod:`utils.ContextLogger`

************************

''' 

__author__ = "cued_dialogue_systems_group"
import ConfusionModel
import NBestGenerator
from utils import Settings
from utils import ContextLogger
logger = ContextLogger.getLogger('')


class EM(object):
    '''
    Main class for error simulation

    :param None:
    '''
    def __init__(self, domainString):        
        
        # DEFAULTS:
        self.nBestSize = 1
        self.confusionModelName = 'RandomConfusions'
        self.nBestGeneratorName = 'UniformNBestGenerator'
        #self.nBestGeneratorName = 'DSTC2NBestGenerator'
        
        # CONFIG:
        if Settings.config.has_option('em', 'nbestsize'):
            self.nBestSize = Settings.config.getint('em','nbestsize')
        if Settings.config.has_option('em','confusionmodel'):
            self.confusionModelName = Settings.config.get('em','confusionmodel')
        if Settings.config.has_option('em','nbestgeneratormodel'):
            self.nBestGeneratorName = Settings.config.get('em','nbestgeneratormodel')
        
        logger.info('N-best list size: '+str(self.nBestSize))
        logger.info('N-best generator model: '+self.nBestGeneratorName)
        logger.info('Confusion model: '+self.confusionModelName)

        # Create confusion model.
        if self.confusionModelName == 'RandomConfusions':
            self.confusionModel = ConfusionModel.EMRandomConfusionModel(domainString)
        else:
            logger.error('Confusion model '+self.confusionModelName+' is not implemented.')
        
        # Create N-best generator.
        if self.nBestGeneratorName == 'UniformNBestGenerator':
            self.nBestGenerator = NBestGenerator.EMNBestGenerator(self.confusionModel, self.nBestSize)
        elif self.nBestGeneratorName == 'SampledNBestGenerator':
            logger.warning('Note the original C++ implementation of EMSampledNBestGenerator was actually the same to EMUniformNBestGenerator.')
            logger.warning('Here the size of N-best list is also sampled from uniform distribution of [1,..,N].')
            self.nBestGenerator = NBestGenerator.EMSampledNBestGenerator(self.confusionModel, self.nBestSize)
        elif self.nBestGeneratorName == 'DSTC2NBestGenerator':
            self.nBestGenerator = NBestGenerator.DSTC2NBestGenerator(self.confusionModel, self.nBestSize)
        else:
            logger.error('N-best generator '+self.nBestGeneratorName+' is not implemented.')
     
    def setErrorRate(self, err):
        '''
        :param err: error rate
        :type err: int
        :returns: None
        '''
        self.nBestGenerator.set_error_rate(err)
        self.errorRate = err
    
    def getNBest(self, a_u):
        '''
        Returns a list of simulated semantic hypotheses given the true act a_u

        :param a_u: None
        :type a_u: str
        :returns: (instance) 
        '''
        return self.nBestGenerator.getNBest(a_u)




# END OF FILE
