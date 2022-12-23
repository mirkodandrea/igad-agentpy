from .constants import (
    EMIT_EARLY_WARNING_PROBABILITY, MAX_YEARS
)

import agentpy as ap
import numpy as np
from numpy.random import random
from .households import HouseHold


class Model(ap.Model):
    def setup(self):
        n_households = self.p['positions']
        p = self.p

        households = map(
            lambda i: HouseHold(
                self,
                position=p['positions'][i],
                price=p['prices'][i],
                income=p['incomes'][i],
                vulnerability=p['vulnerabilities'][i],
                family_members=p['family_members'][i],
                awareness=p['awarenesses'][i],
                fear=p['fears'][i]
            ),
            range(len(n_households))
        )
        self.households = ap.AgentList(self, households, HouseHold)
        self.domain = ap.Space(self, [180]*2, torus=True)
        self.domain.add_agents(self.households, positions=self.p['positions'])

    def emit_early_warning(self):
        """ 
        emit early warning if there is a flood event in 
        current year using EMIT_EARLY_WARNING_PROBABILITY
        """
        t = self.t
        if t not in self.p.events:
            return False

        return random() < EMIT_EARLY_WARNING_PROBABILITY

    def step(self):
        """ Call a method for every agent. """
        previous_awareness = [h.awareness for h in self.households]
        previous_perception = [h.perception for h in self.households]

        if self.emit_early_warning():
            self.households.receive_early_warning(previous_perception)

        self.households.update_awareness(previous_awareness)
        self.households.update_fear()

        if self.t in self.p.events:
            # there is one or more event this year
            flood_values = self.households.do_flood()
            self.households.set_fear(flood_values)
            self.households.displace()
        else:
            # no event this year
            self.households.fix_damage()

    def update(self):
        """ Record a dynamic variable. """
        self.households.record('damage')
        self.households.record('perception')
        self.households.record('awareness')
        self.households.record('fear')
        self.households.record('displaced')

        if (self.t + 1) >= MAX_YEARS:
            self.stop()

    def end(self):
        """ Repord an evaluation measure. """
        #self.report('income', 1)
        pass
