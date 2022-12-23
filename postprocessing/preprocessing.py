#%%
import pandas as pd
import geopandas as gpd
import rasterio as rio
# Read the data

bbox = gpd.read_file('data_2/bbox.gpkg')

#%%
rivers = gpd.read_file('data_2/rivers_bbox.gpkg')
#%%
_settlements = gpd.read_file('data_2/settlements_grid.gpkg')
xmin, ymin, xmax, ymax = 453624.99, 1779456.63,  460131.58, 1784763.63
stl = _settlements.cx[xmin:xmax, ymin:ymax].iloc[::10]
#%%
stl.plot()
#%%
stores = gpd.read_file('data_2/food_stores.gpkg')
others = gpd.read_file('data_2/others.gpkg')
roads = gpd.read_file('data_2/roads_bbox.gpkg')


#%%
stl['roads_dst'] = stl.distance(roads.unary_union)
stl['roads_rank'] = 1 - stl['roads_dst'].rank(pct = True)
stl['stores_dst'] = stl.distance(stores.unary_union)
stl['stores_rank'] = 1 - stl['stores_dst'].rank(pct = True)
stl['others_dst'] = stl.distance(others.unary_union)
stl['others_rank'] = 1 - stl['others_dst'].rank(pct=True)

#settlements['rivers_dst'] = settlements.distance(rivers.unary_union)

#%%

price = stl['roads_rank'] + stl['stores_rank'] + stl['others_rank']
stl['price'] = price
#%%


stl.to_file('data_2/settlements_with_price.gpkg', driver = 'GPKG')
# %%
