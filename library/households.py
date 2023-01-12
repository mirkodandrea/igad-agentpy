from .constants import (
    POVERTY_LINE, MAX_DISTANCE, FLOOD_DAMAGE_THRESHOLD, 
    FLOOD_DAMAGE_MAX, FLOOD_FEAR_MAX,
    DISPLACE_DAMAGE_THRESHOLD,
    
    EVENT_FLOOD, EVENT_EARLY_WARNING
)

import agentpy as ap
import numpy as np
from numpy.random import normal, random

class HouseHold(ap.Agent):
    def setup(self, 
        position, 
        income, 
        price, 
        vulnerability, 
        family_members, 
        awareness, 
        fear,
        trust
        ):
        self.position = position
        self.income = income
        self.price = price
        self.vulnerability = vulnerability
        self.family_members = family_members
        self.awareness = awareness
        self.fear = fear
        self.trust = trust
        self.damage = 0
        self.displaced = False
        self.reason = ''
        
        self.last_life_events = []

    def update_trust(self):
        if EVENT_FLOOD in self.last_life_events and EVENT_EARLY_WARNING in self.last_life_events:
            self.trust = 1.0
            pass

        elif EVENT_FLOOD not in self.last_life_events and EVENT_EARLY_WARNING in self.last_life_events:
            self.trust = np.clip(self.trust/2, 0, 1)
            pass

        elif EVENT_FLOOD in self.last_life_events and EVENT_EARLY_WARNING not in self.last_life_events:
            # do nothing
            pass
        
        elif EVENT_FLOOD not in self.last_life_events and EVENT_EARLY_WARNING not in self.last_life_events:
            self.trust = np.clip(self.trust * 1.1, 0, 1)
            pass

        self.last_life_events = []


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

        if self.damage < 0.25 and self.displaced:
            self.displaced = False

    def update_awareness(self, previous_awareness):
        """
        update risk awareness for current household
        """
        neighbors = self.model.domain.neighbors(self, distance=MAX_DISTANCE)
        if len(neighbors) == 0:
            return

        neighbour_awareness = [previous_awareness[n.id-1] for n in neighbors]
        self.awareness = normal(np.mean(neighbour_awareness), 0.2)

    def do_flood(self, events):
        """check flood for current household
        increment damage if household is flooded
        - damage is in percentage
        - damage occurs if flood value is > 10mm
        - damage is proportional to flood value: 1m -> 100% damage
        """
        flood_value = np.NaN

        for event in events:
            r = event['rio_object']
            flood_data = event['data']
            row, col = r.index(*self.position)
            flood_value = np.nanmax(
                [flood_data[row, col],
                 flood_value]
            )

        # flood value is in millimeters
        if flood_value > FLOOD_DAMAGE_THRESHOLD:
            self.damage = np.clip(
                self.damage + (flood_value / FLOOD_DAMAGE_MAX), 0, 1)
            
        if flood_value > 0:
            self.last_life_events.append(EVENT_FLOOD)

        return flood_value

    @property
    def perception(self):
        return self.awareness * self.fear

    def displace(self):
        """
        decide if household is displaced
        """
        if self.displaced:
            return

        # if household income is below poverty line, it cannot be displaced
        if self.damage >= DISPLACE_DAMAGE_THRESHOLD:
            self.__set_displaced()
        elif self.damage >= 0.1:
            self.displaced = normal(self.damage, 0.2) < self.perception
        else:
            self.displaced = False

    def update_fear(self):
        t = self.model.t - 1
        if t == 0:
            return

        fear_recovery = 0.2
        self.fear = np.clip(self.fear - fear_recovery, 0, 1)
    
    def set_fear(self, flood_values):
        """
        update fear for current household
        """
        flood_value = flood_values[self.id-1]
        if flood_value > 0:
            self.fear = flood_values[self.id-1] / FLOOD_FEAR_MAX

    def __set_displaced(self):
        self.displaced = True

    def receive_early_warning(self, previous_perception):
        """
        receive early warning and decide if household is displaced
        """
        self.last_life_events.append(EVENT_EARLY_WARNING)

        if self.displaced:
            return

        if self.income <= POVERTY_LINE:
            # if household income is below poverty line, 
            # it cannot displaced by early warning
            return


        # check trust in authorities
        if random() < self.trust:
            self.__set_displaced()


        # neighbors = self.model.domain.neighbors(self, distance=MAX_DISTANCE)
        # if len(neighbors) == 0:
        #     return

        # neighbours_perception = [previous_perception[n.id-1] for n in neighbors]
        # # if at least 50% of neighbours has a perception > 0.5
        # # then household is displaced
        # if np.mean(neighbours_perception) > 0.5:
        #     self.__set_displaced(REASON_EARLY_WARNING)

            