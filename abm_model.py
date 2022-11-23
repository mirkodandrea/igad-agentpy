#%%
import agentpy as ap
import contextily as cx
import geopandas as gpd
import IPython
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio

from numpy.random import random, normal, poisson, pareto
#%%
# parametro per scalare la distribuzione dell'income (crescente -> distribuizione più piatta)
ALPHA_INCOME = 0.5
# parametro di povertà
POVERTY_LINE = 2
# 1KM 
MAX_DISTANCE = 0.002
# minimum flood depth to be considered as a flood
FLOOD_DAMAGE_THRESHOLD = 10
FLOOD_DAMAGE_MAX = 1000
FLOOD_FEAR_MAX = 300

#%%
def get_events(initial_row=0, stride=20):
    events = []
    df_floods = pd.read_csv('data_2/IGAD/SD_EventCalendar.csv')
    for idx, row in df_floods.iloc[initial_row:stride].iterrows():
        with rio.open(f'data_2/IGAD/Maps/SD_30mHazardMap_{row.EventID:0>4d}.tif') as f:
            flood_data = f.read(1)

        events.append(
            dict(
                data=flood_data, 
                year=row.Year, 
                interarrival_time=row.InterarrivalTime,
                rio_object=f
        ))
    
    return events
# %%
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

    def fix_damage(self):
        """
        fix damage for current household
        use interarrival_time to calculate how much damage can be fixed
        """
        t = self.model.t -1
        if t==0: return
        
        interarrival_time = self.p.events[t]['interarrival_time']

        # it takes 1 year to fix 100% of damage if income is >=10
        # else every unit of income less than 10 increases the time to fix the damage by 1.2 month
        recovery_time = 365  \
                        if self.income > 10 \
                        else 365 + (10 - self.income) * 36.5

        recovery = interarrival_time/recovery_time
        
        self.damage = np.clip(self.damage - recovery, 0, 1)

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
        t = self.model.t - 1
        event = self.p.events[t]
        r = event['rio_object']
        flood_data = event['data']
        row, col = r.index(*self.position)
        flood_value = flood_data[row, col]

        # flood value is in millimeters
        if flood_value > FLOOD_DAMAGE_THRESHOLD:
            self.damage = np.clip(self.damage + (flood_value / FLOOD_DAMAGE_MAX), 0, 1)

        return flood_value

    @property
    def perception(self):
        return self.awareness * self.fear

    def displace(self):
        """
        decide if household is displaced
        """
        if self.displaced: return

        # if household income is below poverty line, it cannot be displaced
        if self.income <= POVERTY_LINE:
            self.displaced = False
            return

        if self.damage >= 0.65:
            self.displaced = True
        elif self.damage >= 0.1:
            self.displaced = normal(self.damage, 0.2) < self.perception
        else:
            self.displaced = False

    def update_fear(self):
        t = self.model.t -1
        if t==0: return
        
        interarrival_time = self.p.events[t]['interarrival_time']
        fear_recovery = interarrival_time/365        
        self.fear = np.clip(self.damage - fear_recovery, 0, 1)


    def set_fear(self, flood_values):
        """
        update fear for current household
        """
        flood_value = flood_values[self.id-1]
        if flood_value > 0:
            self.fear = flood_values[self.id-1] / FLOOD_FEAR_MAX

        
#%%
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


    def step(self):
        """ Call a method for every agent. """
        previous_awareness = [h.awareness for h in self.households]

        self.households.fix_damage()
        self.households.update_awareness(previous_awareness)
        self.households.update_fear()
        flood_values = self.households.do_flood()        
        self.households.set_fear(flood_values)
        self.households.displace()


    def update(self):
        """ Record a dynamic variable. """
        self.households.record('damage')
        self.households.record('perception')
        self.households.record('awareness')
        self.households.record('fear')
        self.households.record('displaced')
        
        if (self.t + 1) >= len(self.p.events):
            self.stop()

    def end(self):
        """ Repord an evaluation measure. """
        #self.report('income', 1)
        pass

#%%
def animation_plot_single(m, ax):
    print(f"{m.t}")
    ax.set_title(f"t={m.t}")
    displaced_idx = np.array(m.households.displaced)

    pos = m.domain.positions.values()
    pos = np.array(list(pos)).T  
    
    damages = np.array(m.households.damage)

    ax.scatter(*pos[:, ~displaced_idx], c=damages[~displaced_idx],
               cmap='viridis', marker='s', s=3 + 25*awarenesses[~displaced_idx])
    ax.scatter(*pos[:, displaced_idx], c=damages[displaced_idx], cmap='viridis', marker='x', s=3 + 25*awarenesses[displaced_idx])

    cx.add_basemap(ax, crs='epsg:4326',
                   source=cx.providers.OpenStreetMap.Mapnik)
    #ax.set_xlim(0, m.p.size)
    #ax.set_ylim(0, m.p.size)
    ax.set_axis_off()
        
def animation_plot(model):    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection=None)
    animation = ap.animate(model, fig, ax, animation_plot_single)
    return IPython.display.HTML(animation.to_jshtml(fps=3)) 

# %%



settlements = gpd.read_file('data_2/settlements_with_price.gpkg').to_crs(epsg=4326)
events = get_events(stride=30)
n_households = len(settlements)
prices = settlements['price'].values
lons = settlements.geometry.centroid.x
lats = settlements.geometry.centroid.y
positions = list(zip(lons, lats))
incomes = pareto(ALPHA_INCOME, n_households)
vulnerabilities = random(n_households)
awarenesses = random(n_households)
fears = random(n_households)
family_members = poisson(4, n_households)+1
params = dict(
    positions=positions,
    prices=prices,
    incomes=incomes,
    vulnerabilities=vulnerabilities,
    family_members=family_members,
    events=events,
    awarenesses=awarenesses,
    fears=fears
)
# %%
model = Model(params)
animation_plot(model)
#%%
df = model.output['variables']['HouseHold']
df['damage'].unstack().mean(axis=1).plot()
#%%
df['displaced'].unstack().mean(axis=1).plot()
# %%
df['awareness'].unstack().mean(axis=1).plot()
# %%
df['fear'].unstack().mean(axis=1).plot()
# %%
df['perception'].unstack().mean(axis=1).plot()
# %%
#sometimes_displaced = df['displaced'].groupby('obj_id').any()
