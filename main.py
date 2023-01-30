# %%
# !wget apps.cimafoundation.org/html/IGAD.zip -O IGAD.zip
# !unzip IGAD.zip
# %pip install -r requirements.txt

# %%
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

# %%
def animation_plot_single(m, ax):
    print(f"{m.t}")
    ax.set_title(f"t={m.t}")
    displaced_idx = np.array([s != 'normal' for s in m.households.status])
    perceptions = np.array(m.households.perception)

    poors_idx = np.array(m.households.income) < POVERTY_LINE

    pos = m.domain.positions.values()
    pos = np.array(list(pos)).T

    damages = np.array(m.households.damage)
    awarenesses = np.array(m.households.awareness)

    ax.scatter(*pos[:, ~displaced_idx], c=damages[~displaced_idx],
               cmap='viridis', marker='s', s=3 + 25*perceptions[~displaced_idx])
    ax.scatter(*pos[:, displaced_idx], c=damages[displaced_idx],
               cmap='viridis', marker='x', s=3 + 25*perceptions[displaced_idx])

    ax.scatter(*pos[:, poors_idx], facecolors='none',
               edgecolors='g', marker='o', s=30)

    cx.add_basemap(ax, crs='epsg:4326',
                   source=cx.providers.OpenStreetMap.Mapnik)
    #ax.set_xlim(0, m.p.size)
    #ax.set_ylim(0, m.p.size)
    ax.set_axis_off()


def animation_plot(model):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection=None)
    animation = ap.animate(model, fig, ax, animation_plot_single)
    return IPython.display.HTML(animation.to_jshtml(fps=3))


# %%

settlements = gpd.read_file(
    'IGAD/settlements_with_price.gpkg').to_crs(epsg=4326)
events = get_events(initial_year=0, stride=MAX_YEARS)
n_households = len(settlements)

lons = settlements.geometry.centroid.x
lats = settlements.geometry.centroid.y
flood_prones = (lons - lons.min()) / (lons.max() - lons.min()) < 0.3
positions = list(zip(lons, lats))
incomes = pareto(ALPHA_INCOME, n_households)

awarenesses = random(n_households)
fears = random(n_households)
trusts = random(n_households)


false_alarm_rate = 0.3
false_negative_rate = 0.0

params = dict(
    positions=positions,
    trusts=trusts,
    incomes=incomes,
    flood_prones=flood_prones,
    events=events,
    awarenesses=awarenesses,
    fears=fears,
    false_alarm_rate=false_alarm_rate,
    false_negative_rate=false_negative_rate,
)
# %%
model = Model(params)
animation_plot(model)


# %%

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

plot_stats(model, 'awareness')
plot_stats(model, 'fear')
plot_stats(model, 'perception')
plot_stats(model, 'trust')


plot_stats(model, 'trust')



