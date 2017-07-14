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
ConfusionModel.py - handcrafted SemI error creator 
===================================================

Copyright CUED Dialogue Systems Group 2015 - 2017

.. seealso:: CUED Imports/Dependencies: 

    import :mod:`utils.DiaAct` |.|
    import :mod:`ontology.Ontology` |.|
    import :mod:`utils.Settings` |.|
    import :mod:`utils.ContextLogger`

************************

''' 

__author__ = "cued_dialogue_systems_group"
import copy

from utils import Settings
from utils import DiaAct
from ontology import Ontology
from utils import ContextLogger
logger = ContextLogger.getLogger('')


class EMConfusionModel(object):
    '''Base class for EMRandomConfusionModel. 

        .. Note:: 
            Used through derived class only. 
    '''
    def create_wrong_hyp(self, a_u):
        '''Create a wrong hypothesis for a_u

        :param a_u: of :class:`DiaAct`
        :type a_u: instance
        :returns: (instance) of :class:`DiaAct` - modified input act
        '''
        confact_is_same = True
        num_attempts = 0
        max_num_attempts = 25
        conf_act = None
        while confact_is_same and num_attempts < max_num_attempts:
            conf_act = self.confuse_hyp(a_u)
            confact_is_same = (conf_act == a_u)
            num_attempts += 1

        if num_attempts == max_num_attempts:
            logger.warning("Confused act same after %d attempts: return null() instead." % max_num_attempts)
            return DiaAct.DiaAct('null()')

        return conf_act


class EMRandomConfusionModel(EMConfusionModel):
    '''Derived class from :class:`EMConfusionModel`.

    :param None:
    '''
    def __init__(self, domainString):
        self.domainString = domainString

        self.CONFUSE_TYPE = 0.2
        self.CONFUSE_SLOT = 0.3
        self.CONFUSE_VALUE = 0.5
        self.newUserActs = ['hello',
                            'thankyou',
                            'ack',
                            'bye',
                            'inform',
                            'request',
                            'reqalts',
                            'reqmore',
                            'confirm',
                            'affirm',
                            'negate',
                            'deny',
                            'repeat',
                            'null']
        self.nNewUserActs = len(self.newUserActs)
    
    def confuse_hyp(self, a_u):
        '''Randomly confuse the act type, slot or value. 

        :param a_u: of :class:`DiaAct`
        :type a_u: instance
        :returns: (instance) of :class:`DiaAct` - modified input act
        '''
        wHyp = copy.deepcopy(a_u)
        
        # Identify if this diaact type takes 0, 1, or 2 arguments
        nSlotVal = wHyp.getDiaItemFormat()
        
        # Make a choice to confuse the type, slot or value
        choice = Settings.random.choice([0, 1, 2], p=[self.CONFUSE_TYPE, self.CONFUSE_SLOT, self.CONFUSE_VALUE])
        choice = min(choice, nSlotVal)
        
        if choice == 0:
            wHyp = self._confuse_type(wHyp)
        elif choice == 1:
            wHyp = self._confuse_slot(wHyp)
        elif choice == 2:
            wHyp = self._confuse_value(wHyp)
        else:
            logger.error('Invalid choice '+str(choice))        
        
        return wHyp
    
    def _confuse_dia_act_type(self, oldtype):
        '''
        Randomly select a dialogue act type different from oldtype.
        '''
        acttypeset = copy.copy(self.newUserActs)
        acttypeset.remove(oldtype)
        return Settings.random.choice(acttypeset)

    def _confuse_slot_name(self, old_name):
        '''
        Randomly select a slot name that is different from the given old_name
        '''
        slots = Ontology.global_ontology.get_requestable_slots(self.domainString) 
        if old_name in slots:
            slots.remove(old_name)
        # if old_name not in slots:
        #     logger.error('Slot "%s" is not found in ontology.' % old_name)

        return Settings.random.choice(slots)

    

    def _get_confused_value_for_slot(self, slot, old_val):
        '''
        Randomly select a slot value for the given slot s different from old_val.
        '''
        return Ontology.global_ontology.getRandomValueForSlot(self.domainString, slot=slot, notthese=[old_val])
            
    def _confuse_type(self, hyp):
        '''
        Create a wrong hypothesis, where the dialogue act type is different.
        '''
        hyp.items = []
        hyp.act = self._confuse_dia_act_type(hyp.act)
        item_format = DiaAct.actTypeToItemFormat[hyp.act]
        if item_format == 0:
            return hyp
        elif item_format == 1:
            new_slot_name = Ontology.global_ontology.get_random_slot_name(self.domainString)
            hyp.append(new_slot_name, None)
        elif item_format == 2:
            new_slot_name = Ontology.global_ontology.get_random_slot_name(self.domainString)
            assert new_slot_name is not None
            new_slot_val = Ontology.global_ontology.getRandomValueForSlot(self.domainString, slot=new_slot_name)
            hyp.append(new_slot_name, new_slot_val)
        # TODO: If item_format is 3, it doesn't confuse slot-values.
        # This might be a bug in the original implementation.
        return hyp
    
    def _confuse_slot(self, hyp):
        '''
        Create a wrong hypothesis, where the slot names are different.
        '''
        for dip in hyp.items:
            # If the slot is empty, just break
            if dip.slot is None:
                break
            
            slot = dip.slot
            if slot == 'more':
                break

            dip.slot = self._confuse_slot_name(slot)
            if dip.val is not None:
                dip.val = Ontology.global_ontology.getRandomValueForSlot(self.domainString, slot=dip.slot)
        
        return hyp

    def _confuse_value(self, a_u):
        '''
        Create a wrong hypothesis, where one slot value is different.
        '''
        rand = Settings.random.randint(len(a_u.items))
        a_u_i = a_u.items[rand]
        
        if a_u_i.slot is not None and a_u_i.val is not None:
            a_u.items[rand].val = self._get_confused_value_for_slot(a_u_i.slot, a_u_i.val)
        
        return a_u

#END OF FILE
