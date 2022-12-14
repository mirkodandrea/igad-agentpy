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
POVERTY_LINE = 1
# 1KM 
MAX_DISTANCE = 0.002
# minimum flood depth to be considered as a flood
FLOOD_DAMAGE_THRESHOLD = 10
FLOOD_DAMAGE_MAX = 1000
FLOOD_FEAR_MAX = 300

MAX_YEARS = 100

#%%
def get_events(initial_year, stride):
    events = {}
    df_floods = pd.read_csv('data_2/IGAD/SD_EventCalendar.csv')
    df_floods = df_floods.query('Year >= @initial_year and Year < @initial_year+@stride')

    for year, group in df_floods.groupby('Year'):
        for idx, row in group.iterrows():
            with rio.open(f'data_2/IGAD/Maps/SD_30mHazardMap_{row.EventID:0>4d}.tif') as f:
                flood_data = f.read(1)

            if year not in events:
                events[year] = []

            events[year - initial_year].append(
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
        """
        t = self.model.t -1
        if t==0: return

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
        if self.damage >= 0.65:
            self.displaced = True
        elif self.damage >= 0.1:
            self.displaced = normal(self.damage, 0.2) < self.perception
        else:
            self.displaced = False

    def update_fear(self):
        t = self.model.t -1
        if t==0: return
        
        fear_recovery = 0.2
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

#%%
def animation_plot_single(m, ax):
    print(f"{m.t}")
    ax.set_title(f"t={m.t}")
    displaced_idx = np.array(m.households.displaced)
    poors_idx = np.array(m.households.income) < POVERTY_LINE

    pos = m.domain.positions.values()
    pos = np.array(list(pos)).T  
    
    damages = np.array(m.households.damage)

    ax.scatter(*pos[:, ~displaced_idx], c=damages[~displaced_idx],
               cmap='viridis', marker='s', s=3 + 25*awarenesses[~displaced_idx])
    ax.scatter(*pos[:, displaced_idx], c=damages[displaced_idx], cmap='viridis', marker='x', s=3 + 25*awarenesses[displaced_idx])

    ax.scatter(*pos[:, poors_idx], facecolors='none',
               edgecolors='g', marker='o', s=30)

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
events = get_events(initial_year=0, stride=MAX_YEARS)
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
def plot_stats(model, variable):
    df = model.output['variables']['HouseHold']
    var = df[variable].unstack().mean(axis=0)
    ax = var.plot.bar()
    max_y = var.max()*1.1
    ax.set_ylim(0, max_y)
    events_years = events.keys()
    ax.stem(events_years, [max_y]*len(events_years), 'r')
    ax.set_title(variable)
    plt.show()


plot_stats(model, 'damage')
plot_stats(model, 'displaced')
plot_stats(model, 'awareness')
plot_stats(model, 'fear')
plot_stats(model, 'perception')


# %%
