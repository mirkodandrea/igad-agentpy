from .constants import (
    POVERTY_LINE, MAX_DISTANCE, FLOOD_DAMAGE_THRESHOLD, 
    FLOOD_DAMAGE_MAX, FLOOD_FEAR_MAX,
    DISPLACE_DAMAGE_THRESHOLD,
    
    EVENT_FLOOD, EVENT_EARLY_WARNING
)

STATUS_NORMAL = 'normal'
STATUS_EVACUATED = 'evacuated'
STATUS_DISPLACED = 'displaced'

import agentpy as ap
import numpy as np
from numpy.random import normal, random

class HouseHold(ap.Agent):
    def setup(self, 
        position, 
        income, 
#        price, 
#        vulnerability, 
#        family_members, 
        flood_prone,
        awareness, 
        fear,
        trust
        ):
        self.position = position
        self.income = income
        # self.price = price
        # self.vulnerability = vulnerability
        # self.family_members = family_members
        self.awareness = awareness
        self.flood_prone = flood_prone
        self.fear = fear
        self.trust = trust
        self.damage = 0
        self.displaced = False
        
        self.status = STATUS_NORMAL
        
        # prepared to flood
        self.alerted = False
        self.received_flood = False
        self.prepared = False

    def init_step(self):
        """init household status for each step"""
        # set all flags back to False
        self.alerted = False
        self.received_flood = False

        # check household damage and decide to return to normal
        pass

        if self.status == STATUS_EVACUATED:
            self.prepared = False

    def receive_early_warning(self):
        """receive early warning from government
        - if household is already evacuated, do nothing
        - if household doesn't trust the government, then do nothing
        - if household is not aware of risk, prepare
        - if household is aware of risk, evacuate without preparing
        """
        self.alerted = True

        if self.status != STATUS_NORMAL:
            # already evacuated
            return
        
        if self.income < POVERTY_LINE:
            # poor household
            return
        
        if self.trust >= 0.5:
            # trust the government
            if self.perception >= 0.5:
                # aware of risk
                self.status = STATUS_EVACUATED
                return

            self.prepared = True
        
            

    def check_neighbours(self):
        """check neighbours for early warning reaction
        - if enough neighbours are evacuated, then evacuate
        - if enough neighbours are prepared, then prepare
        """
        neighbours = self.model.domain.neighbors(self, MAX_DISTANCE)
        
        other_prepared = [neighbour.prepared for neighbour in neighbours]
        if sum(other_prepared) > 0.5 * len(neighbours):
            self.prepared = True

        other_status = [neighbour.status == STATUS_EVACUATED for neighbour in neighbours]
        if sum(other_status) > 0.5 * len(neighbours):
            self.status = STATUS_EVACUATED

            
    def receive_flood(self, flood_value):
        """receive flood for current household
        increment damage if household is flooded
        - damage is in percentage
        - damage occurs if flood value is > 10mm
        - damage is proportional to flood value: 1m -> 100% damage
        """

        if flood_value == 0:
            """ nothing happened to this household """
            # [TODO] update trust if alerted
            self.received_flood = False
            return

        self.received_flood = True

        if self.prepared and flood_value < FLOOD_DAMAGE_THRESHOLD:
            return
        
        new_damage = (flood_value / FLOOD_DAMAGE_MAX)
        if self.prepared:
            new_damage = new_damage * 0.5

        self.damage = np.clip(self.damage + new_damage, 0, 1)        
        FLOOD_DAMAGE_AWARENESS_THRESHOLD = 0.1        
        if new_damage > FLOOD_DAMAGE_AWARENESS_THRESHOLD:
            self.awareness = 1.0

        if self.damage >= DISPLACE_DAMAGE_THRESHOLD:
            self.status = STATUS_DISPLACED

    def end_step(self):
        if not self.receive_flood and not self.alerted:
            self.fix_damage()

        self.update_sentiments()

    def update_sentiments(self):
        """
        update trust for current household
        - if household is alerted and at least a neighbour received flood, trust -> 1
        - if household is alerted and no neighbour received flood, trust -> trust/2
        - if household is not alerted and at least a neighbour received flood, fear -> fear + 0.1
        - if household is not alerted and noone received flood, do nothing
        """
        neighbours_flooded = [neighbour.received_flood for neighbour in self.model.domain.neighbors(
            self, MAX_DISTANCE)]
        anyone_flooded = \
            self.received_flood or \
            any(neighbours_flooded)
        
        if self.received_flood:
            self.awareness = 1.0

        elif len(neighbours_flooded) > 0 \
            and sum(neighbours_flooded) > 0.25 * len(neighbours_flooded):
            self.awareness = 1.0

        if self.alerted and anyone_flooded:
            self.trust = 1.0
        
        elif self.alerted and not anyone_flooded:
            self.trust = np.clip(self.trust/2, 0, 1)

        elif not self.alerted and anyone_flooded:
            self.fear = np.clip(self.fear + 0.1, 0, 1)
            pass
        
        elif not self.alerted and not anyone_flooded:
            pass

    def fix_damage(self):
        """
        fix damage for current household
        """
        t = self.model.t - 1
        if t == 0:
            return

        if self.income <= POVERTY_LINE:
            return

        if self.displaced:
            # it takes 3 year to fix 100% of damage if income is >=10
            recovery_time = 3*365
        else:
            recovery_time = 365

        # every unit of income less than 10 increases the time to fix the damage by 6 month
        if self.income < 10:
            recovery_time += (10 - self.income) * 180

        recovery = 365/recovery_time

        self.damage = np.clip(self.damage - recovery, 0, 1)

        if self.damage < 0.25 and self.status != STATUS_NORMAL:
            self.status = STATUS_NORMAL

    
    @property
    def perception(self):
        return self.awareness * self.fear

    
    