#%%
import agentpy as ap
import contextily as cx
import geopandas as gpd
import IPython
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
#%%
# parametro per scalare la distribuzione dell'income (crescente -> distribuizione più piatta)
ALPHA_INCOME = 0.5
# parametro di povertà
POVERTY_LINE = 2
#%% 1KM 
MAX_DISTANCE = 0.005
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
    @staticmethod
    def create(model, *, positions, prices, incomes, vulnerabilities, family_members, perceptions):
        households = []
        for position, price, income, vulnerability, members, perception in zip(positions, prices, incomes, vulnerabilities, family_members, perceptions):
            households.append(HouseHold(model, position=position, price=price, income=income, vulnerability=vulnerability, family_members=members, perception=perception))

        return households


    def setup(self,**kwargs):
        self.position = kwargs['position']
        self.income = kwargs['income']
        self.price = kwargs['price']
        self.vulnerability = kwargs['vulnerability']
        self.family_members = kwargs['family_members']
        self.perception = kwargs['perception']
        self.damage = 0
        self.displaced = False


    def step(self):
        self.fix_damage()
        flood_value = self.do_flood()
        self.displace()
        self.update_perception(flood_value)
        

    def displace(self):
        """
        check if household is displaced
        """
        if self.displaced: return

        # if household income is below poverty line, it cannot be displaced
        if self.income <= POVERTY_LINE:
            self.displaced = False
            return

        if self.damage >= 0.65:
            self.displaced = True
        elif self.damage >= 0.1:
            self.displaced = np.random.random() < self.perception
        else:
            self.displaced = False

    def fix_damage(self):
        """
        fix damage for current household
        use interarrival_time to calculate how much damage can be fixed
        """
        t = self.model.t -1
        if t==0: return
        
        interarrival_time = self.p.events[t]['interarrival_time']
        recovery_time = 365  \
                        if self.income > 10 \
                        else 365 + (10 - self.income) * 36.5

        recovery = interarrival_time/recovery_time
        
        self.damage = np.clip(self.damage - recovery, 0, 1)


    def do_flood(self):
        """check flood for current household
        increment damage if household is flooded
        """
        t = self.model.t - 1
        event = self.p.events[t]
        r = event['rio_object']
        flood_data = event['data']
        row, col = r.index(*self.position)
        flood_value = flood_data[row, col]

        self.damage = np.clip(self.damage + ((flood_value - 10)/1000), 0, 1)

        return flood_value

    def __get_max_perception(self):
        # [TODO] max_perception should be a function of agent characteristics
        return 1

    def update_perception(self, flood_value):
        """
        update risk perception for current household
        """
        if flood_value > 0.01:
            self.perception = self.__get_max_perception()
            return
        
        
        neighbors = self.model.domain.neighbors(self, distance=MAX_DISTANCE)
        if len(neighbors) == 0:
            #self.perception = np.clip(self.perception - 0.1, 0, 1)
            return
        
        high_perception_neighbours = \
            list(filter(
                lambda n: n.perception > self.perception, 
            neighbors))
        
        if len(high_perception_neighbours) > len(neighbors)/4:
            self.perception = np.clip(self.perception + 0.1, 0, 1)

    #------------------------------------------------------------------------
        # Define custom actions here
        #print('agent', self.id, 'is doing something')
        # 

        
        # mean_income = np.mean([n.income for n in neighbors])
        # if self.income > mean_income*1.5:
        #     # find agent with highest income
        #     richest_household = max(neighbors, key=lambda n: n.income)

        #     self.move(richest_household)
        # else:
        #     self.move()

    # def move(self, target=None):
    #     if target is None:
    #         new_x = int(np.random.random() * 3 - 1.5)
    #         new_y = int(np.random.random() * 3 - 1.5)
    #     else:
    #         position = self.model.domain.positions[self]
    #         target_position = self.model.domain.positions[target]
    #         new_x = target_position[0] - position[0]
    #         new_y = target_position[1] - position[1]
    #         module = np.sqrt(new_x**2 + new_y**2)
    #         if module > 0:
    #             new_x = (new_x/module)
    #             new_y = (new_y/module)

    #     self.model.domain.move_by(self, (new_x, new_y))

    # def move_toward(self, target):
    #     # Move towards target
    #     pass
        
#%%
class Model(ap.Model):
    def setup(self):
        """ Initiate a list of new households. """
        households = HouseHold.create(self, 
            positions=self.p['positions'], 
            prices=self.p['prices'], 
            incomes=self.p['incomes'], 
            vulnerabilities=self.p['vulnerabilities'], 
            family_members=self.p['family_members'],
            perceptions=self.p['perceptions']
        )
        self.households = ap.AgentList(self, households, HouseHold)
        self.domain = ap.Space(self, [180]*2, torus=True)
        self.domain.add_agents(self.households, positions=self.p['positions'])


    def step(self):
        """ Call a method for every agent. """
        self.households.step()

    def update(self):
        """ Record a dynamic variable. """
        self.households.record('damage')
        self.households.record('displaced')
        
        if (self.t + 1) >= len(self.p.events):
            self.stop()

    def end(self):
        """ Repord an evaluation measure. """
        self.report('income', 1)

#%%
def animation_plot_single(m, ax):
    ax.set_title(f"t={m.t}")
    displaced_idx = np.array(m.households.displaced)

    pos = m.domain.positions.values()
    pos = np.array(list(pos)).T  
    
    damages = np.array(m.households.damage)

    ax.scatter(*pos[:, ~displaced_idx], c=damages[~displaced_idx], cmap='viridis', marker='s', s=20)
    ax.scatter(*pos[:, displaced_idx], c=damages[displaced_idx], cmap='viridis', marker='x', s=20)

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
events = get_events()
n_households = len(settlements)
prices = settlements['price'].values
lons = settlements.geometry.centroid.x
lats = settlements.geometry.centroid.y
positions = list(zip(lons, lats))
incomes = np.random.pareto(ALPHA_INCOME, n_households)
vulnerabilities = np.random.random(n_households)
perceptions = np.random.random(n_households)
family_members = np.random.poisson(4, n_households)+1
params = dict(
    positions=positions,
    prices=prices,
    incomes=incomes,
    vulnerabilities=vulnerabilities,
    perceptions=perceptions,
    family_members=family_members,
    events=events
)
# %%
model = Model(params)
animation_plot(model)
#%%
damage_df = model.output['variables']['HouseHold']
#%%
damage_df.unstack().sum(axis=0).plot()

# %%
