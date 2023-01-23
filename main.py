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

from library.households import HouseHold
from library.model import Model
from library.utils import get_events
from library.constants import (
    POVERTY_LINE, MAX_YEARS, ALPHA_INCOME
)



#%%
def animation_plot_single(m, ax):
    print(f"{m.t}")
    ax.set_title(f"t={m.t}")
    displaced_idx = np.array([s != 'normal' for s in m.households.status])
    perceptions = np.array(m.households.perception)
    poors_idx = np.array(m.households.income) < POVERTY_LINE

    pos = m.domain.positions.values()
    pos = np.array(list(pos)).T  
    
    damages = np.array(m.households.damage)

    ax.scatter(*pos[:, ~displaced_idx], c=damages[~displaced_idx],
               cmap='viridis', 
               marker='s', 
               s=3 + 25*perceptions[~displaced_idx])
    
    ax.scatter(*pos[:, displaced_idx], c=damages[displaced_idx], 
                cmap='viridis', 
                marker='x', 
                s=3 + 25*perceptions[displaced_idx])

    ax.scatter(*pos[:, poors_idx], facecolors='none',
               edgecolors='g', marker='o', s=30)

    cx.add_basemap(ax, crs='epsg:4326',
                   source=cx.providers.OpenStreetMap.Mapnik)
    
    ax.set_axis_off()
        
def animation_plot(model):    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection=None)
    animation = ap.animate(model, fig, ax, animation_plot_single)
    return IPython.display.HTML(animation.to_jshtml(fps=3)) 

# %%



settlements = gpd.read_file('IGAD/settlements_with_price.gpkg').to_crs(epsg=4326)
events = get_events(initial_year=0, stride=MAX_YEARS)
n_households = len(settlements)
lons = settlements.geometry.centroid.x
lats = settlements.geometry.centroid.y
positions = list(zip(lons, lats))
incomes = pareto(ALPHA_INCOME, n_households)
awarenesses = random(n_households)
fears = random(n_households)

#prices = settlements['price'].values
#vulnerabilities = random(n_households)
#family_members = poisson(4, n_households)+1

params = dict(
    positions=positions,
    incomes=incomes,
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
