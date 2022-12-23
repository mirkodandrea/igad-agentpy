from .constants import (
    POVERTY_LINE, MAX_DISTANCE, FLOOD_DAMAGE_THRESHOLD, 
    FLOOD_DAMAGE_MAX, FLOOD_FEAR_MAX,
    DISPLACE_DAMAGE_THRESHOLD
)

import agentpy as ap
import numpy as np
from numpy.random import normal

class HouseHold(ap.Agent):
    def setup(self, position, income, price, vulnerability, family_members, awareness, fear):
        self.position = position
        self.income = income
        self.price = price
        self.vulnerability = vulnerability
        self.family_members = family_members
        self.awareness = awareness
        self.fear = fear
        self.damage = 0
        self.displaced = False
        self.reason = ''

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

    def do_flood(self):
        """check flood for current household
        increment damage if household is flooded
        - damage is in percentage
        - damage occurs if flood value is > 10mm
        - damage is proportional to flood value: 1m -> 100% damage
        """
        t = self.model.t
        events = self.p.events[t]
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
            self.__set_displaced('damage')
        elif self.damage >= 0.1:
            self.displaced = normal(self.damage, 0.2) < self.perception
        else:
            self.displaced = False

    def update_fear(self):
        t = self.model.t - 1
        if t == 0:
            return

        fear_recovery = 0.2
        self.fear = np.clip(self.damage - fear_recovery, 0, 1)

    def set_fear(self, flood_values):
        """
        update fear for current household
        """
        flood_value = flood_values[self.id-1]
        if flood_value > 0:
            self.fear = flood_values[self.id-1] / FLOOD_FEAR_MAX

    def __set_displaced(self, reason):
        self.displaced = True
        self.reason = reason

    def receive_early_warning(self, previous_perception):
        """
        receive early warning and decide if household is displaced
        """
        if self.displaced:
            return

        if self.income <= POVERTY_LINE:
            # if household income is below poverty line, 
            # it cannot displaced by early warning
            return

        # check neighbours perception and if at least 50% of neighbours has a perception > 0.5
        # then household is displaced
        if self.perception > 0.5:
            self.__set_displaced('early warning')

        neighbors = self.model.domain.neighbors(self, distance=MAX_DISTANCE)
        if len(neighbors) == 0:
            return

        neighbours_perception = [previous_perception[n.id-1] for n in neighbors]
        # if at least 50% of neighbours has a perception > 0.5
        # then household is displaced
        if np.mean(neighbours_perception) > 0.5:
            self.__set_displaced('early warning')

            