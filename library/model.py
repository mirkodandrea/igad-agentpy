from .constants import (
    MAX_YEARS, 
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
                flood_prone=p['flood_prones'][i],
                awareness=p['awarenesses'][i],
                fear=p['fears'][i],
                trust=p['trusts'][i]
            ),
            range(len(n_households))
        )
        self.false_alarm_rate = self.p['false_alarm_rate']
        self.false_negative_rate = self.p['false_negative_rate']

        self.households = ap.AgentList(self, households, HouseHold)
        self.domain = ap.Space(self, [180]*2, torus=True)
        self.domain.add_agents(self.households, positions=self.p['positions'])

    def maybe_emit_early_warning(self):
        """ 
        emit early warning.
        If there is a flood event in time t, emit early warning with probability 1 - false_negative_rate
        If there is no flood event in time t, emit early warning with probability false_alarm_rate        
        """
        t = self.t
        emit = False
        if t not in self.p.events:
            emit = random() < self.false_alarm_rate
        else:
            emit = random() < self.false_negative_rate

        if emit:
            self.households.receive_early_warning()

    def step(self):
        """ Call a method for every agent. """
        
        self.maybe_emit_early_warning()

        if self.t in self.p.events:
            events = self.p.events[self.t]
            self.do_flood(events)
        else:
            # no event this year
            self.nothing_happened()

    def do_flood(self, events):
        """ 
        do flood
        """
        for household in self.households:
            flood_value = 0
            for event in events:
                r = event['rio_object']
                flood_data = event['data']
                row, col = r.index(*household.position)
                flood_value = np.nanmax(
                    [flood_data[row, col],
                    flood_value]
                )
            household.receive_flood(flood_value)

    def notify_nothing_happened(self):
        """ 
        notify nothing happened
        """
        self.households.nothing_happened()
        

    def update(self):
        """ Record a dynamic variable. """
        self.households.record('damage')
        self.households.record('perception')
        self.households.record('awareness')
        self.households.record('fear')
        self.households.record('displaced')
        self.households.record('trust')

        if (self.t + 1) >= MAX_YEARS:
            self.stop()

    def end(self):
        """ Repord an evaluation measure. """
        #self.report('income', 1)
        pass
