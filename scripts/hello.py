#%%
from xview import utils
import xarray as xr
import pandas as pd
import numpy as np
import shapely
import geopandas as gpd
# %%
ds = xr.open_dataset("https://thredds.niva.no/thredds/dodsC/datasets/nrt/color_fantasy.nc")
# %%
ds.sel(time=slice()
#%%
idx
#%%
points = shapely.points((10.459527, 54.540728))
#%%
p = shapely.from_wkt("POINT (10.459527 54.540728)")
#%%
mp = shapely.from_wkt("MULTIPOINT ((10.459527 54.540728), (10.49454 54.540728))")
# %%
gdf = gpd.GeoDataFrame(
    {"geometry": [shapely.points(ds["longitude"].values, ds["latitude"].values)]},
)
# %%

gdf = gpd.GeoDataFrame(
    [], geometry=gpd.points_from_xy(ds.longitude, ds.latitude), crs="EPSG:4326"
)# %%

# %%
ds.isel(time=gdf[gdf.geometry==shapely.points(10.459527, 54.540728)].index)
# %%
pol = shapely.Polygon([(10.459527, 54.540728), (10.49454, 54.540728), (10.49454, 54.554007), (10.459527, 54.554007)])
pol
#%%
idx_gdf = gdf[gdf.geometry.within(pol)]

# %%
ds_idx = ds.set_xindex(["longitude", "latitude"])


# %%
mp = shapely.from_wkt("MULTIPOINT ((10.459527 54.540728), (10.49454 54.540728), (10.474357 54.545358))")
#%%
for p in mp.coords:
    print(p)
# %%
points = []
for point in mp.geoms:
    try:
        points.append(ds_idx.sel(longitude=point.x, latitude=point.y).indexs)
    except KeyError:
        pass
# %%
ds.sel(time=[ds_idx.sel(longitude=point.x, latitude=point.y)])

